# -*- coding: utf-8 -*-

import heapq
import random


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def is_empty(self):
        return not self._queue

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))        
        self._index += 1

    def pop(self):
        if self.is_empty():
            raise IndexError('Pop from empty PriorityQueue')
        return heapq.heappop(self._queue)

    def peek(self, k):
        if self.is_empty():
            return []
        return heapq.nsmallest(k, self._queue)

    def size(self):
        return len(self._queue)
    
    def random_elements(self, num_elements):
        if self.is_empty():
            raise IndexError('Random from empty PriorityQueue')
        if num_elements > len(self._queue):
            num_elements = len(self._queue)
        random_indices = random.sample(range(len(self._queue)), num_elements)
        return [self._queue[i][-1] for i in random_indices]

