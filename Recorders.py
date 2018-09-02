# -*- coding: utf-8 -*-
"""
Use ZMQ to receice data and save data

data format:
index,
dataMatrix

dataMatix:
columns represent data group
lines   represent measurements


# ==============================================================================
# Author: jmcui
# Date:   2016-5-30
# Mail:   jmcui@mail.ustc.edu.cn
# ==============================================================================
"""

import threading
import time
import sys
import os
import zmq
from IonControlerSocket import SerializingContext

if sys.version_info < (3, 4):
    import cPickle as pickle
else:
    import _pickle as pickle
import gzip
# import ipdb
# ipdb.set_trace()


class PickleRecorder(object):
    '''  A Recorder use python pickle
    '''
    Type = 'Pickle'
    # __metaclass__ = ABCMeta

    def __init__(self, title, port=b'tcp://localhost:5566',
                 fpath='.',
                 fname='untitle',
                 surfix='.dat',
                 ftime=True,
                 zmq_ctx=None,
                 ):
        '''
        title: the head or title in the data file
        to label the data or explain the data

        '''
        if zmq_ctx is None:
            self.ctx = SerializingContext()
        else:
            self.ctx = zmq_ctx
        self.sub = self.ctx.socket(zmq.SUB)
        self.port = port
        self.sub.connect(self.port)
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')  # to subcribe all message
        self.i = 0
        self.title = title

        if ftime:
            filetime = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
        else:
            filetime = ''
        # fname = os.path.join(fpath, fname + filetime + surfix)
        fname = os.path.join(fpath, fname + surfix)
        self.f = gzip.open(fname, 'wb')
        self.running = False

    def run(self):
        ''' Run task
        creat a thread to run loop function
        '''

        if self.running:
            print("Recorder %s is running" % self.Title)
            return
        self.running = True

        # first dump the head (title) of the data
        pickle.dump(self.title, self.f, protocol=2)

        def loop_func(self):
            while(self.running):
                self.loop()

        self.thd = threading.Thread(target=loop_func, args=[self])

        self._x0 = time.time()
        self.thd.start()

    def loop(self):
        ''' loop for getting data thread
        '''
        idn, data = self.sub.recv_pickle()
        t = time.time() - self._x0
        self.i += 1
        pickle.dump((t, idn, data), self.f, protocol=2)

    def stop(self):
        self.running = False  # stop loop
        if self.thd is not None:
            self.thd.join()
        self.f.flush()
        self.f.close()

    def __del__(self):
        self.sub.close()
        self.ctx.term()
        self.f.close()


class PickleEventRecorder(object):
    '''  A Recorder use python pickle
    '''
    Type = 'Pickle'
    # __metaclass__ = ABCMeta

    def __init__(self, title, port='tcp://localhost:5566',
                 fpath='.',
                 fname='untitle',
                 surfix='.dat',
                 ftime=True,
                 zmq_ctx=None,
                 ):
        '''
        title: the head or title in the data file
        to label the data or explain the data

        '''
        if zmq_ctx is None:
            self.ctx = SerializingContext()
        else:
            self.ctx = zmq_ctx
        self.sub = self.ctx.socket(zmq.SUB)
        self.port = port
        self.sub.connect(self.port)
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')  # to subcribe all message
        self.i = 0
        self.title = title

        if ftime:
            filetime = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
        else:
            filetime = ''
        fname = os.path.join(fpath, fname + filetime + surfix)
        self.f = gzip.open(fname, 'wb')
        self.running = False

    def run(self):
        ''' Run task
        creat a thread to run loop function
        '''

        if self.running:
            print("Recorder %s is running" % self.Title)
            return
        self.running = True

        # first dump the head (title) of the data
        pickle.dump(self.title, self.f, protocol=2)

        def loop_func(self):
            while(self.running):
                self.loop()

        self.thd = threading.Thread(target=loop_func, args=[self])

        self._x0 = time.time()
        self.thd.start()

    def loop(self):
        ''' loop for getting data thread
        '''
        idn, data = self.sub.recv_pickle()
        t = time.time() - self._x0
        self.i += 1
        pickle.dump((t, idn, data), self.f, protocol=2)

    def stop(self):
        self.running = False  # stop loop
        if self.thd is not None:
            self.thd.join()
        # self.f.flush()
        self.f.close()

    def __del__(self):
        self.sub.close()
        self.ctx.term()
        self.f.close()

if __name__ == '__main__':
    test = PickleRecorder('test')
