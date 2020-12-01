"""
NWBIO
========

IO class for reading data from a Neurodata Without Borders (NWB) dataset

Documentation : https://neurodatawithoutborders.github.io
Depends on: h5py, nwb, dateutil
Supported: Read, Write
Specification - https://github.com/NeurodataWithoutBorders/specification
Python APIs - (1) https://github.com/AllenInstitute/nwb-api/tree/master/ainwb
	          (2) https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/core/nwb_data_set.py
              (3) https://github.com/NeurodataWithoutBorders/api-python
Sample datasets from CRCNS - https://crcns.org/NWB
Sample datasets from Allen Institute - http://alleninstitute.github.io/AllenSDK/cell_types.html#neurodata-without-borders
"""

from __future__ import absolute_import, division
from neo.core import baseneo

import os
from itertools import chain
from datetime import datetime
import json
try:
    from json.decoder import JSONDecodeError
except ImportError:  # Python 2
    JSONDecodeError = ValueError
from collections import defaultdict

import numpy as np
import quantities as pq
from siunits import *
from neo.io.baseio import BaseIO
from neo.io.proxyobjects import (
    AnalogSignalProxy as BaseAnalogSignalProxy,
    EventProxy as BaseEventProxy,
    EpochProxy as BaseEpochProxy,
    SpikeTrainProxy as BaseSpikeTrainProxy
)
from neo.core import (Segment, SpikeTrain, Unit, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, ChannelIndex, Block, ImageSequence)

# PyNWB imports
try:
    import pynwb
    from pynwb import NWBFile, TimeSeries, get_manager
    from pynwb.base import ProcessingModule
    from pynwb.ecephys import ElectricalSeries, Device, EventDetection
    from pynwb.icephys import VoltageClampSeries, VoltageClampStimulusSeries, CurrentClampStimulusSeries, CurrentClampSeries, PatchClampSeries, SweepTable
    from pynwb.behavior import SpatialSeries
    from pynwb.misc import AnnotationSeries
    from pynwb import image
    from pynwb.image import ImageSeries
    from pynwb.spec import NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBNamespace, NWBNamespaceBuilder
    from pynwb.device import Device
    # For calcium imaging data
    from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence
    have_pynwb = True
except ImportError:
    have_pynwb = False
except SyntaxError:  # pynwb doesn't support Python 2.7
    have_pynwb = False

import random

# hdmf imports
try:
    from hdmf.spec import (LinkSpec, GroupSpec, DatasetSpec, SpecNamespace,
                           NamespaceBuilder, AttributeSpec, DtypeSpec, RefSpec)
    have_hdmf = True
except ImportError:
    have_hdmf = False
except SyntaxError:
    have_hdmf = False


GLOBAL_ANNOTATIONS = (
    "session_start_time", "identifier", "timestamps_reference_time", "experimenter",
    "experiment_description", "session_id", "institution", "keywords", "notes",
    "pharmacology", "protocol", "related_publications", "slices", "source_script",
    "source_script_file_name", "data_collection", "surgery", "virus", "stimulus_notes",
    "lab", "session_description",
)

POSSIBLE_JSON_FIELDS = (
    "source_script", "description"
)

prefix_map = {
    1e9: 'giga',
    1e6: 'mega',
    1e3: 'kilo',
    1: '',
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano',
    1e-12: 'pico'
}

def try_json_field(content):
    """
    Try to interpret a string as JSON data.

    If successful, return the JSON data (dict or list)
    If unsuccessful, return the original string
    """
    try:
        return json.loads(content)
    except JSONDecodeError:
        return content


def get_class(module, name):
    """
    Given a module path and a class name, return the class object
    """
    module_path = module.split(".")
    assert len(module_path) == 2  # todo: handle the general case where this isn't 2
    return getattr(getattr(pynwb, module_path[1]), name)


def statistics(block):  # todo: move this to be a property of Block
    """
    Return simple statistics about a Neo Block.
    """
    stats = {
        "SpikeTrain": {"count": 0},
        "AnalogSignal": {"count": 0},
        "IrregularlySampledSignal": {"count": 0},
        "Epoch": {"count": 0},
        "Event": {"count": 0},
    }
    for segment in block.segments:
        stats["SpikeTrain"]["count"] += len(segment.spiketrains)
        stats["AnalogSignal"]["count"] += len(segment.analogsignals)
        stats["IrregularlySampledSignal"]["count"] += len(segment.irregularlysampledsignals)
        stats["Epoch"]["count"] += len(segment.epochs)
        stats["Event"]["count"] += len(segment.events)
###        stats["ImageSequence"]["count"] += len(segment.imagesequence)
    return stats


def get_units_conversion(signal, timeseries_class):
    """
    Given a quantity array and a TimeSeries subclass, return
    the conversion factor and the expected units
    """
    # it would be nice if the expected units was an attribute of the PyNWB class
    if "CurrentClamp" in timeseries_class.__name__:
        expected_units = pq.volt
    elif "VoltageClamp" in timeseries_class.__name__:
        expected_units = pq.ampere
    else:
        # todo: warn that we don't handle this subclass yet
        expected_units = signal.units
    return float((signal.units/expected_units).simplified.magnitude), expected_units


def time_in_seconds(t):
    return float(t.rescale("second"))


def _decompose_unit(unit):
    """
    Given a quantities unit object, return a base unit name and a conversion factor.

    Example:

    >>> _decompose_unit(pq.mV)
    ('volt', 0.001)
    """
    assert isinstance(unit, pq.quantity.Quantity)
    assert unit.magnitude == 1
    conversion = 1.0

    def _decompose(unit):
        dim = unit.dimensionality
        if len(dim) != 1:
            raise NotImplementedError("Compound units not yet supported")  # e.g. volt-metre
        uq, n = list(dim.items())[0]
        if n != 1:
            raise NotImplementedError("Compound units not yet supported")  # e.g. volt^2
        uq_def = uq.definition
        return float(uq_def.magnitude), uq_def
    conv, unit2 = _decompose(unit)
    while conv != 1:
        conversion *= conv
        unit = unit2
        conv, unit2 = _decompose(unit)
    return list(unit.dimensionality.keys())[0].name, conversion


def _recompose_unit(base_unit_name, conversion):
    """
    Given a base unit name and a conversion factor, return a quantities unit object

    Example:

    >>> _recompose_unit("ampere", 1e-9)
    UnitCurrent('nanoampere', 0.001 * uA, 'nA')

    """
    if conversion not in prefix_map:
        raise ValueError(f"Can't handle this conversion factor: {conversion}")
    unit_name = prefix_map[conversion] + base_unit_name
    if unit_name[-1] == "s":  # strip trailing 's', e.g. "volts" --> "volt"
        unit_name = unit_name[:-1]
    return getattr(pq, unit_name)


class NWBIO(BaseIO):
    """
    Class for "reading" experimental data from a .nwb file, and "writing" a .nwb file from Neo
    """
    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                         SpikeTrain, Epoch, Event, ImageSequence]
    readable_objects = supported_objects
    writeable_objects = supported_objects

    has_header = False
    support_lazy = True

    name = 'NeoNWB IO'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'one-file'

    is_readable = True
    is_writable = True
    is_streameable = False

    def __init__(self, filename, mode='r'):
        """
        Arguments:
            filename : the filename
        """
        if not have_pynwb:
            raise Exception("Please install the pynwb package to use NWBIO")
        if not have_hdmf:
            raise Exception("Please install the hdmf package to use NWBIO")
        BaseIO.__init__(self, filename=filename)
        self.filename = filename
        self.blocks_written = 0
        self.nwb_file_mode = mode

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_all_blocks(self, lazy=False, **kwargs):
        """

        """
        assert self.nwb_file_mode in ('r',)
        io = pynwb.NWBHDF5IO(self.filename, mode=self.nwb_file_mode)  # Open a file with NWBHDF5IO
        self._file = io.read()

        self.global_block_metadata = {}
        for annotation_name in GLOBAL_ANNOTATIONS:
            value = getattr(self._file, annotation_name, None)
            if value is not None:
                if annotation_name in POSSIBLE_JSON_FIELDS:
                    value = try_json_field(value)
                self.global_block_metadata[annotation_name] = value
        if "session_description" in self.global_block_metadata:
            self.global_block_metadata["description"] = self.global_block_metadata["session_description"]
        self.global_block_metadata["file_origin"] = self.filename
        if "session_start_time" in self.global_block_metadata:
            self.global_block_metadata["rec_datetime"] = self.global_block_metadata["session_start_time"]
        if "file_create_date" in self.global_block_metadata:
            self.global_block_metadata["file_datetime"] = self.global_block_metadata["file_create_date"]

        self._blocks = {}
        self._read_acquisition_group(lazy=lazy)
        self._read_stimulus_group(lazy)
        self._read_units(lazy=lazy)
        self._read_epochs_group(lazy)

        return list(self._blocks.values())

    def read_block(self, lazy=False, **kargs):
        """
        Load the first block in the file.
        """
        return self.read_all_blocks(lazy=lazy)[0]

    def _get_segment(self, block_name, segment_name):
        # If we've already created a Block with the given name return it,
        #   otherwise create it now and store it in self._blocks.
        # If we've already created a Segment in the given block, return it,
        #   otherwise create it now and return it.
        if block_name in self._blocks:
            block = self._blocks[block_name]
        else:
            block = Block(name=block_name, **self.global_block_metadata)
            self._blocks[block_name] = block
        segment = None
        for seg in block.segments:
            if segment_name == seg.name:
                segment = seg
                break
        if segment is None:
            segment = Segment(name=segment_name)
            segment.block = block
            block.segments.append(segment)     
        return segment


    def _read_epochs_group(self, lazy):
        if self._file.epochs is not None:
            try:
                # NWB files created by Neo store the segment, block and epoch names as extra columns
                segment_names = self._file.epochs.segment[:]
                block_names = self._file.epochs.block[:]
                epoch_names = self._file.epochs._name[:]
            except AttributeError:
                epoch_names = None

            if epoch_names is not None:
                unique_epoch_names = np.unique(epoch_names)
                for epoch_name in unique_epoch_names:
                    index = (epoch_names == epoch_name)
                    epoch = EpochProxy(self._file.epochs, epoch_name, index)
                    if not lazy:
                        epoch = epoch.load()
                    segment_name = np.unique(segment_names[index])
                    block_name = np.unique(block_names[index])
                    assert segment_name.size == block_name.size == 1
                    segment = self._get_segment(block_name[0], segment_name[0])
                    segment.epochs.append(epoch)
                    epoch.segment = segment
            else:
                epoch = EpochProxy(self._file.epochs)
                if not lazy:
                    epoch = epoch.load()
                segment = self._get_segment("default", "default")
                segment.epochs.append(epoch)
                epoch.segment = segment
    
    def _read_timeseries_group(self, group_name, lazy):
        group = getattr(self._file, group_name)
        for timeseries in group.values():     
            try:
                # NWB files created by Neo store the segment and block names in the comments field
                hierarchy = json.loads(timeseries.comments)
            except JSONDecodeError:
                # For NWB files created with other applications, we put everything in a single
                # segment in a single block
                # todo: investigate whether there is a reliable way to create multiple segments,
                #       e.g. using Trial information
                block_name = "default"
                segment_name = "default"
            else:
                block_name = hierarchy["block"]
                segment_name = hierarchy["segment"]

            segment = self._get_segment(block_name, segment_name)
            if isinstance(timeseries, AnnotationSeries):
                event = EventProxy(timeseries, group_name)
                if not lazy:
                    event = event.load()
                segment.events.append(event)
                event.segment = segment
            elif timeseries.rate:  # AnalogSignal
                signal = AnalogSignalProxy(timeseries, group_name)
                if not lazy:
                    signal = signal.load()
                segment.analogsignals.append(signal)
                signal.segment = segment
            else:  # IrregularlySampledSignal
                signal = AnalogSignalProxy(timeseries, group_name)
                if not lazy:
                    signal = signal.load()
                segment.irregularlysampledsignals.append(signal)
                signal.segment = segment

    def _read_units(self, lazy):
        if self._file.units:
            for id in self._file.units.id[:]:
                try:
                    # NWB files created by Neo store the segment and block names as extra columns
                    segment_name = self._file.units.segment[id]
                    block_name = self._file.units.block[id]
                except AttributeError:
                    # For NWB files created with other applications, we put everything in a single
                    # segment in a single block
                    segment_name = "default"
                    block_name = "default"
                segment = self._get_segment(block_name, segment_name)
                spiketrain = SpikeTrainProxy(self._file.units, id)
                if not lazy:
                    spiketrain = spiketrain.load()
                segment.spiketrains.append(spiketrain)
                spiketrain.segment = segment

    def _read_acquisition_group(self, lazy):
        self._read_timeseries_group("acquisition", lazy)

    def _read_stimulus_group(self, lazy):
        self._read_timeseries_group("stimulus", lazy)

    def write_all_blocks(self, blocks, **kwargs):
        """
        Write list of blocks to the file
        """
        # todo: allow metadata in NWBFile constructor to be taken from kwargs
        start_time = datetime.now()
        annotations = defaultdict(set)
        for annotation_name in GLOBAL_ANNOTATIONS:
            if annotation_name in kwargs:
                annotations[annotation_name] = kwargs[annotation_name]
            else:
                for block in blocks:
                    if annotation_name in block.annotations:
                        try:
                            annotations[annotation_name].add(block.annotations[annotation_name])
                        except TypeError:
                            if annotation_name in POSSIBLE_JSON_FIELDS:
                                encoded = json.dumps(block.annotations[annotation_name])
                                annotations[annotation_name].add(encoded)
                            else:
                                raise
                if annotation_name in annotations:
                    if len(annotations[annotation_name]) > 1:
                        raise NotImplementedError(
                            "We don't yet support multiple values for {}".format(annotation_name))
                    # take single value from set
                    annotations[annotation_name], = annotations[annotation_name]
        if "identifier" not in annotations:
            annotations["identifier"] = self.filename
        if "session_description" not in annotations:
            annotations["session_description"] = blocks[0].description or self.filename
            # todo: concatenate descriptions of multiple blocks if different
        if "session_start_time" not in annotations:
            annotations["session_start_time"] = datetime.now()
        # todo: handle subject
        # todo: store additional Neo annotations somewhere in NWB file
        nwbfile = NWBFile(**annotations)

        assert self.nwb_file_mode in ('w',)  # possibly expand to 'a'ppend later
        if self.nwb_file_mode == "w" and os.path.exists(self.filename):
            os.remove(self.filename)
        io_nwb = pynwb.NWBHDF5IO(self.filename, manager=get_manager(), mode=self.nwb_file_mode)

        if sum(statistics(block)["SpikeTrain"]["count"] for block in blocks) > 0:
            nwbfile.add_unit_column('_name', 'the name attribute of the SpikeTrain')
            nwbfile.add_unit_column(
                'segment', 'the name of the Neo Segment to which the SpikeTrain belongs')
            nwbfile.add_unit_column(
                'block', 'the name of the Neo Block to which the SpikeTrain belongs')  

        if sum(statistics(block)["Epoch"]["count"] for block in blocks) > 0:
            nwbfile.add_epoch_column('_name', 'the name attribute of the Epoch')
            nwbfile.add_epoch_column(
                'segment', 'the name of the Neo Segment to which the Epoch belongs')
            nwbfile.add_epoch_column('block', 'the name of the Neo Block to which the Epoch belongs')

        for i, block in enumerate(blocks):
            self.write_block(nwbfile, block)
        io_nwb.write(nwbfile)
        io_nwb.close()

    def write_block(self, nwbfile, block, **kwargs):
        """
        Write a Block to the file
            :param block: Block to be written
        """
        electrodes = self._write_electrodes(nwbfile, block)
        if not block.name:
            block.name = "block%d" % self.blocks_written
        for i, segment in enumerate(block.segments):
            assert segment.block is block
            if not segment.name:
                segment.name = "%s : segment%d" % (block.name, i)
            self._write_segment(nwbfile, segment, electrodes)
        self.blocks_written += 1

    def _write_electrodes(self, nwbfile, block):
        # this handles only icephys_electrode for now
        electrodes = {}
        devices = {}
        nwb_sweep_tables = {}
        for segment in block.segments:
            for signal in chain(segment.analogsignals, segment.irregularlysampledsignals):
                if "nwb_electrode" in signal.annotations:
                    elec_meta = signal.annotations["nwb_electrode"].copy()
                    if elec_meta["name"] not in electrodes:
                        # todo: check for consistency if the name is already there
                        if elec_meta["device"]["name"] in devices:
                            device = devices[elec_meta["device"]["name"]]
                        else:
                            device = nwbfile.create_device(**elec_meta["device"])
                            devices[elec_meta["device"]["name"]] = device
                        elec_meta.pop("device")
                        electrodes[elec_meta["name"]] = nwbfile.create_icephys_electrode(
                            device=device, **elec_meta
                        )
        return electrodes

    def _write_segment(self, nwbfile, segment, electrodes):
        # maybe use NWB trials to store Segment metadata?
        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            assert signal.segment is segment
            if not signal.name:
                signal.name = "%s : analogsignal%d" % (segment.name, i)
            self._write_signal(nwbfile, signal, electrodes)

        for i, train in enumerate(segment.spiketrains):
            assert train.segment is segment
            if not train.name:
                train.name = "%s : spiketrain%d" % (segment.name, i)
            self._write_spiketrain(nwbfile, train)            
                
        for i, event in enumerate(segment.events):
            assert event.segment is segment
            if not event.name:                
                event.name = "%s : event%d" % (segment.name, i)
            self._write_event(nwbfile, event)

        for i, epoch in enumerate(segment.epochs):
            if not epoch.name:
                epoch.name = "%s : epoch%d" % (segment.name, i)
            self._write_epoch(nwbfile, epoch)

    def _write_signal(self, nwbfile, signal, electrodes):
        hierarchy = {'block': signal.segment.block.name, 'segment': signal.segment.name}

        if "nwb_type" in signal.annotations:
            timeseries_class = get_class(*signal.annotations["nwb_type"])
        else:
            timeseries_class = TimeSeries  # default

        additional_metadata = {name[4:]: value
                               for name, value in signal.annotations.items()
                               if name.startswith("nwb:")}

        if "nwb_electrode" in signal.annotations:
            electrode_name = signal.annotations["nwb_electrode"]["name"]
            additional_metadata["electrode"] = electrodes[electrode_name]
        
        if "nwb_sweep_number" in signal.annotations:
            sweep_table_name = signal.annotations["nwb_sweep_number"]["name"]

        if timeseries_class != TimeSeries:
            conversion, units = get_units_conversion(signal, timeseries_class)
            additional_metadata["conversion"] = conversion
        else:
            units = signal.units

        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            nwb_sweep_number = signal.annotations.get("nwb_sweep_number", "nwb_type")
            tS = timeseries_class(
                name=signal.name,
                starting_time=time_in_seconds(signal.t_start),
                data=signal,
                unit=units.dimensionality.string,
                rate=float(sampling_rate),
                comments=json.dumps(hierarchy),
                **additional_metadata)
                # todo: try to add array_annotations via "control" attribute

        elif isinstance(signal, IrregularlySampledSignal):
            tS = timeseries_class(
                name=signal.name,
                data=signal,
                unit=units.dimensionality.string,
                timestamps=signal.times.rescale('second').magnitude,
                comments=json.dumps(hierarchy),
                **additional_metadata)
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(
                signal.__class__.__name__))
        nwb_group = signal.annotations.get("nwb_group", "acquisition")
        add_method_map = {
            "acquisition": nwbfile.add_acquisition,
            "stimulus": nwbfile.add_stimulus,
        }
        if nwb_group in add_method_map:
            add_time_series = add_method_map[nwb_group]
        else:
            raise NotImplementedError("NWB group '{}' not yet supported".format(nwb_group))
        add_time_series(tS)
        return tS

    def _write_spiketrain(self, nwbfile, spiketrain):
        nwbfile.add_unit(spike_times=spiketrain.rescale('s').magnitude,
                         obs_intervals=[[float(spiketrain.t_start.rescale('s')),
                                         float(spiketrain.t_stop.rescale('s'))]],
                         _name=spiketrain.name,
                         segment=spiketrain.segment.name,
                         block=spiketrain.segment.block.name
                         )
        # todo: handle annotations (using add_unit_column()?)
        # todo: handle Neo Units
        # todo: handle spike waveforms, if any (see SpikeEventSeries)
        return nwbfile.units

    def _write_event(self, nwbfile, event):
        hierarchy = {'block': event.segment.block.name, 'segment': event.segment.name}
        tS_evt = AnnotationSeries(
                        name=event.name,
                        data=event.labels,
                        timestamps=event.times.rescale('second').magnitude,
                        description=event.description or "",
                        comments=json.dumps(hierarchy))
        nwbfile.add_acquisition(tS_evt)
        return tS_evt

    def _write_epoch(self, nwbfile, epoch):
        for t_start, duration, label in zip(epoch.rescale('s').magnitude,
                                            epoch.durations.rescale('s').magnitude,
                                            epoch.labels):
            nwbfile.add_epoch(t_start, t_start + duration, [label], [],
                             _name=epoch.name,
                              segment=epoch.segment.name,
                              block=epoch.segment.block.name
                              )
        return nwbfile.epochs

    def close(self):
        """
        Closes the open nwb file and resets maps.
        """
        if (hasattr(self, "nwb_file") and self.nwb_file and self.nwb_file.is_open()):
            self.nwb_file.close()
            self.nwb_file = None
            self._neo_map = None
            self._ref_map = None
            self._signal_map = None
            self._view_map = None
            self._block_read_counter = None

    def __del__(self):
        self.close()


class AnalogSignalProxy(BaseAnalogSignalProxy):
    common_metadata_fields = (
        # fields that are the same for all TimeSeries subclasses
        "comments", "description", "unit", "starting_time", "timestamps", "rate",
        "data", "starting_time_unit", "timestamps_unit", "electrode",
    )

    def __init__(self, timeseries, nwb_group):
        self._timeseries = timeseries
        self.units = timeseries.unit
        if timeseries.conversion:
            self.units = _recompose_unit(timeseries.unit, timeseries.conversion)
        if timeseries.starting_time is not None:
            self.t_start = timeseries.starting_time * pq.s  # todo: use timeseries.starting_time_unit
        else:
            self.t_start = timeseries.timestamps[0] * pq.s  # todo: use timeseries.timestamps.unit
        if timeseries.rate:
            self.sampling_rate = timeseries.rate * pq.Hz
        else:
            self.sampling_rate = None
        self.name = timeseries.name
        self.annotations = {"nwb_group": nwb_group}
        self.description = try_json_field(timeseries.description)
        if isinstance(self.description, dict):
            self.annotations["notes"] = self.description
            if "name" in self.annotations:
                self.annotations.pop("name")
            self.description = None
        self.shape = self._timeseries.data.shape
        metadata_fields = list(timeseries.__nwbfields__)
        for field_name in self.__class__.common_metadata_fields:  # already handled
            try:
                metadata_fields.remove(field_name)
            except ValueError:
                pass
        for field_name in metadata_fields:
            value = getattr(timeseries, field_name)
            if value is not None:
                self.annotations[f"nwb:{field_name}"] = value
        self.annotations["nwb_type"] = (
            timeseries.__class__.__module__,
            timeseries.__class__.__name__
        )
        
        if hasattr(timeseries, "electrode"):
            # todo: once the Group class is available, we could add electrode metadata
            #       to a Group containing all signals that share that electrode
            #       This would reduce the amount of redundancy (repeated metadata in every signal)
            electrode_metadata = {"device": {}}
            metadata_fields = list(timeseries.electrode.__class__.__nwbfields__) + ["name"]
            metadata_fields.remove("device")  # needs special handling
            for field_name in metadata_fields:
                value = getattr(timeseries.electrode, field_name)
                if value is not None:
                    electrode_metadata[field_name] = value
            for field_name in timeseries.electrode.device.__class__.__nwbfields__:
                value = getattr(timeseries.electrode.device, field_name)
                if value is not None:
                    electrode_metadata["device"][field_name] = value
            self.annotations["nwb_electrode"] = electrode_metadata
        
        if hasattr(timeseries, "image"):
            print("image")


    def load(self, time_slice=None, strict_slicing=True):
        """
        *Args*:
            :time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire signal.
            :strict_slicing: True by default.
                Control if an error is raised or not when one of the time_slice members
                (t_start or t_stop) is outside the real time range of the segment.
        """
        if time_slice:
            i_start, i_stop, sig_t_start = self._time_slice_indices(time_slice,
                                                                    strict_slicing=strict_slicing)
            signal = self._timeseries.data[i_start: i_stop]
        else:            
            signal = self._timeseries.data[:]
            sig_t_start = self.t_start

        if self.annotations=={'nwb_sweep_number'}:
            sweep_number = self._timeseries.sweep_number
        else:
            sweep_table=None

        if self.sampling_rate is None:
            return IrregularlySampledSignal(
                        self._timeseries.timestamps[:] * pq.s,
                        signal,
                        units=self.units,
                        t_start=sig_t_start,
                        sampling_rate=self.sampling_rate,
                        name=self.name,
                        description=self.description,
                        array_annotations=None,
                        sweep_number=sweep_table,
                        **self.annotations)  # todo: timeseries.control / control_description
        else:
            return AnalogSignal(
                        signal,
                        units=self.units,
                        t_start=sig_t_start,
                        sampling_rate=self.sampling_rate,
                        name=self.name,
                        description=self.description,
                        array_annotations=None,
                        sweep_number=sweep_table,
                        **self.annotations)  # todo: timeseries.control / control_description


class EventProxy(BaseEventProxy):

    def __init__(self, timeseries, nwb_group):
        self._timeseries = timeseries
        self.name = timeseries.name
        self.annotations = {"nwb_group": nwb_group}
        self.description = try_json_field(timeseries.description)

        if isinstance(self.description, dict):
            self.annotations.update(self.description)
            self.description = None
        self.shape = self._timeseries.data.shape

    def load(self, time_slice=None, strict_slicing=True):
        """
        *Args*:
            :time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire signal.
            :strict_slicing: True by default.
                Control if an error is raised or not when one of the time_slice members
                (t_start or t_stop) is outside the real time range of the segment.
        """
        if time_slice:
            raise NotImplementedError("todo")
        else:
            times = self._timeseries.timestamps[:]
            labels = self._timeseries.data[:]
        return Event(times * pq.s,
                     labels=labels,
                     name=self.name,
                     description=self.description,
                     **self.annotations)


class EpochProxy(BaseEpochProxy):

    def __init__(self, epochs_table, epoch_name=None, index=None):
        self._epochs_table = epochs_table
        if index is not None:
            self._index = index
            self.shape = (index.sum(),)
        else:
            self._index = slice(None)
            self.shape = epochs_table.n_rows  # untested, just guessed that n_rows exists
        self.name = epoch_name

    def load(self, time_slice=None, strict_slicing=True):
        """
        *Args*:
            :time_slice: None or tuple of the time slice expressed with quantities.
                            None is all of the intervals.
            :strict_slicing: True by default.
                Control if an error is raised or not when one of the time_slice members
                (t_start or t_stop) is outside the real time range of the segment.
        """
        start_times = self._epochs_table.start_time[self._index]
        stop_times = self._epochs_table.stop_time[self._index]
        durations = stop_times - start_times
        labels = self._epochs_table.tags[self._index]

        return Epoch(times=start_times * pq.s,
                     durations=durations * pq.s,
                     labels=labels,
                     name=self.name)


class SpikeTrainProxy(BaseSpikeTrainProxy):

    def __init__(self,  units_table, id):
        self._units_table = units_table
        self.id = id
        self.units = pq.s
        t_start, t_stop = units_table.get_unit_obs_intervals(id)[0]
        self.t_start = t_start * pq.s
        self.t_stop = t_stop * pq.s
        self.annotations = {"nwb_group": "acquisition"}

        try:
            # NWB files created by Neo store the name as an extra column
            self.name = units_table._name[id]
        except AttributeError:
            self.name = None
        self.shape = None   # no way to get this without reading the data

    def load(self, time_slice=None, strict_slicing=True):
        """
        *Args*:
            :time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire spike train.
            :strict_slicing: True by default.
                Control if an error is raised or not when one of the time_slice members
                (t_start or t_stop) is outside the real time range of the segment.
        """
        interval = None
        if time_slice:
            interval = (float(t) for t in time_slice)  # convert from quantities
        spike_times = self._units_table.get_unit_spike_times(self.id, in_interval=interval)
        self.sweep_number = {"nwb_sweep_number"}
        return SpikeTrain(
                    spike_times * self.units,
                    t_stop=self.t_stop,
                    units=self.units,
                    #sampling_rate=array(1.) * Hz, #
                    t_start=self.t_start,
                    waveforms=None, #
                    left_sweep=None, #
                    name=self.name,
                    file_origin=None, #
                    description=None, #
                    array_annotations=None, #
                    id=self.id, ###
                    **self.annotations)
