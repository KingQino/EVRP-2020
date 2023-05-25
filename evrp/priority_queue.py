# -*- coding: utf-8 -*-

import heapq
import random


class PriorityQueue:
    def __init__(self):
        self._queue = []

    def is_empty(self):
        return not self._queue

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, item))        

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
            return []
        if num_elements > len(self._queue):
            num_elements = len(self._queue)
        random_indices = random.sample(range(len(self._queue)), num_elements)
        return [self._queue[i][-1] for i in random_indices]
    
    def remove_elements(self, num_removed_elements):
        if self.is_empty():
            return []
        removed_elements = heapq.nlargest(num_removed_elements, self._queue)
        
        # Convert list of lists into tuple of tuples
        a = [(priority, tuple(map(tuple, sublist))) for priority, sublist in self._queue]
        b = [(priority, tuple(map(tuple, sublist))) for priority, sublist in removed_elements]

        # Perform set difference operation
        a = list(set(a) - set(b))

        heapq.heapify(a)

        # Convert tuples of tuples back into lists of lists
        self._queue = [(priority, list(map(list, sublist))) for priority, sublist in a]

