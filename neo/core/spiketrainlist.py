# -*- coding: utf-8 -*-
"""
This module implements :class:`SpikeTrainList`, a pseudo-list
which supports a multiplexed representation of spike trains
(all times in a single array, with a second array indicating which
neuron/channel the spike is from).
"""

import numpy as np
from .spiketrain import SpikeTrain


class SpikeTrainList(object):
    """
    docstring needed
    """

    def __init__(self, items=None, segment=None):
        """Initialize self"""
        self._items = items
        self._spike_time_array = None
        self._channel_id_array = None
        self._all_channel_ids = None
        self._spiketrain_metadata = None
        self.segment = segment

    def __iter__(self):
        """Implement iter(self)"""
        if self._items is None:
            self._spiketrains_from_array()
        for item in self._items:
            yield item

    def __getitem__(self, i):
        """x.__getitem__(y) <==> x[y]"""
        if self._items is None:
            self._spiketrains_from_array()
        items = self._items[i]
        if isinstance(items, SpikeTrain):
            return items
        else:
            return SpikeTrainList(items=items)

    def __str__(self):
        """Return str(self)"""
        if self._items is None:
            if self._spike_time_array is None:
                return str([])
            else:
                return "SpikeTrainList containing {} spikes from {} neurons".format(
                    self._spike_time_array.size,
                    self._channel_id_array.size)
        else:
            return str(self._items)

    def __len__(self):
        """Return len(self)"""
        if self._items is None:
            if self._all_channel_ids is not None:
                return len(self._all_channel_ids)
            elif self._channel_id_array is not None:
                return np.unique(self._channel_id_array).size
            else:
                return 0
        else:
            return len(self._items)

    def _add_spiketrainlists(self, other):
        if self._spike_time_array is None or other._spike_time_array is None:
            # if either self or other is not storing multiplexed spike trains
            # we return
            return self.__class__(items=self._items[:] + other._items)
        else:
            # both self and other are storing multiplexed spike trains
            # todo: update self._spike_time_array, etc.
            if self._spiketrain_metadata['units'] != other._spiketrain_metadata['units']:
                raise ValueError("Incompatible units")
                # todo: rescale other to the units of self, rather than raising a ValueError
            if self._spiketrain_metadata['t_start'] != other._spiketrain_metadata['t_start']:
                raise ValueError("Incompatible t_start")
                # todo: adjust times and t_start of other to be compatible with self
            if self._spiketrain_metadata['t_stop'] != other._spiketrain_metadata['t_stop']:
                raise ValueError("Incompatible t_stop")
                # todo: adjust t_stop of self and other as necessary
            return self.__class__.from_spike_time_array(
                np.hstack((self._spike_time_array, other._spike_time_array)),
                np.hstack((self._channel_id_array, other._channel_id_array)),
                self._all_channel_ids + other._all_channel_ids,
                units=self._spiketrain_metadata['units'],
                t_start=self._spiketrain_metadata['t_start'],
                t_stop=self._spiketrain_metadata['t_stop'])

    def __add__(self, other):
        """Return self + other"""
        if isinstance(other, self.__class__):
            return self._add_spiketrainlists(other)
        elif other and isinstance(other[0], SpikeTrain):
            for obj in other:
                obj.segment = self.segment
            self._items.extend(other)
            return self
        else:
            return self._items + other

    def __radd__(self, other):
        """Return other + self"""
        if isinstance(other, self.__class__):
            return other._add_spiketrainlists(self)
        elif other and isinstance(other[0], SpikeTrain):
            for obj in other:
                obj.segment = self.segment
            self._items.extend(other)
            return self
        else:
            return other + self._items

    def append(self, obj):
        """L.append(object) -> None -- append object to end"""
        if not isinstance(obj, SpikeTrain):
            raise ValueError("Can only append SpikeTrain objects")
        if self._items is None:
            self._spiketrains_from_array()
        obj.segment = self.segment
        self._items.append(obj)

    def extend(self, iterable):
        """L.extend(iterable) -> None -- extend list by appending elements from the iterable"""
        if self._items is None:
            self._spiketrains_from_array()
        for obj in iterable:
            obj.segment = self.segment
        self._items.extend(iterable)

    @classmethod
    def from_spike_time_array(cls, spike_time_array, channel_id_array,
                              all_channel_ids=None, units='ms',
                              t_start=None, t_stop=None):
        """Create a SpikeTrainList object from an array of spike times
        and an array of channel ids."""
        obj = cls()
        obj._spike_time_array = spike_time_array
        obj._channel_id_array = channel_id_array
        obj._all_channel_ids = all_channel_ids
        obj._spiketrain_metadata = {
            "units": units,
            "t_start": t_start,
            "t_stop": t_stop
        }
        return obj

    def _spiketrains_from_array(self):
        """Convert multiplexed spike time data into a list of SpikeTrain objects"""
        if self._spike_time_array is None:
            self._items = []
        else:
            if self._all_channel_ids is None:
                all_channel_ids = np.unique(self._channel_id_array)
            else:
                all_channel_ids = self._all_channel_ids
            for channel_id in all_channel_ids:
                mask = self._channel_id_array == channel_id
                times = self._spike_time_array[mask]
                spiketrain = SpikeTrain(times, **self._spiketrain_metadata)
                # todo: consider adding channel id as metadata
                spiketrain.segment = self.segment
                self._items.append(spiketrain)

    @property
    def multiplexed(self):
        """Return spike trains as a pair of arrays.

        The first array contains the ids of the channels/neurons that produced each spike,
        the second array contains the times of the spikes.
        """
        if self._spike_time_array is None:
            # need to convert list of SpikeTrains into multiplexed spike times array
            if self._items is None:
                return np.array([]), np.array([])
            else:
                channel_ids = []
                spike_times = []
                for i, spiketrain in enumerate(self._items):
                    spike_times.append(spiketrain.times)
                    channel_ids.append(i * np.ones_like(spiketrain))
                    # todo: what if the spiketrain has stored its channel id as metadata?
                self._spike_time_array = np.hstack(spike_times)
                self._channel_id_array = np.hstack(channel_ids)
        return self._channel_id_array, self._spike_time_array
