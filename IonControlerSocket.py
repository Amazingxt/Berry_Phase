# -*- coding: utf-8 -*-
"""
# Ipython Interface with IonControl

使用两个Socket与IonControl进行数据交互

## zmq 设置

###  端口配置

* 5556 端口  （cmd_port） 
应答模式与IonControl进行命令交互。Ipython notebook 传入命令， IonControl返回数据的方式进行交互。


* 5566 端口  （data_port）
订阅模式， 接收IonControl发出的粗数据


### 命令端口  cmd_port

#### 命令格式：
* 传入格式：
使用pickle 传入一个元组。基本形式是：
       (cmd, arg)
其中 cmd 为命令， arg 为命令参数

cmd所支持的命令有：（参见IonControl中 cmd_loop 函数）

* 传输格式：
       （status，information）
其中 status 为返回状态， True 或者 False， 用来表记指令是否成功执行； information 为具体描述信息，一般为字符串

Created on Aug 21, 2017

@author: jmcui
"""

import zlib
import pickle
import numpy
import zmq


# command port, set command and get response data (REQ and REP mode)
cmd_default = 'tcp://222.195.68.69:5556'
# data port to get raw dataset
data_default = 'tcp://222.195.68.69:5566'


class TaskSocket():
    '''
    cmd and data sockets in pack
    '''

    def __init__(self, cmd_port=cmd_default,
                 data_port=data_default,
                 cmd_timeout=100000,
                 data_timeout=100000
                 ):
        self.ctx = SerializingContext()
        self.req = self.ctx.socket(zmq.REQ)
        self.req.setsockopt(zmq.RCVTIMEO, cmd_timeout)
        self.req.connect(cmd_port)

        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.RCVTIMEO, data_timeout)
        self.sub.connect(data_port)
        # to subcribe all message
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        
        self.cmd_port = cmd_port
        self.data_port = data_port

    def getPlotData(self, PlotName, timeout=100000):
        '''
        To read data from Plotters

        timeout parameter is needed, in ms unit
        For example time=30000, means waiting data out of 30 second,
        the function will return with data None

        '''
        self.req.setsockopt(zmq.RCVTIMEO, timeout)
        self.req.send_pickle(('getPlotData', PlotName))
        try:
            x, y = self.req.recv_pickle()
            y = numpy.array(y)
        except Exception as e:
            x = None  # data None
            y = None
            print(e)
        return (x, y)

    def getRawData(self):

        i, dat = self.sub.recv_pickle()
        return (i, numpy.array(dat))
    
    def setCode(self, code, task='Remote'):
        ''' Load task, and then set code in the task
        code: the code to set
        task: task name to load and set, default is remote 
        '''
        return self.cmd('setCode', code, task)
   
    def func(self, *args):
        ''' execute task function
        * args[0], should be task function name
        * other args is function args 
        '''
        return self.cmd('TaskFun', *args)
    
    def loadTask(self, *args):
        '''load Task in Gui Controller
        * args[0], should be task  name
        '''
        return self.cmd('loadTask', *args)

    def cmd(self, command, *args):
        '''
        send command to IonController

        commmand + argument
        '''
 
        self.req.send_pickle((command, args))
        try:
            result = self.req.recv_pickle()
        except Exception as e:
            result = None
            print(e)
        return result

    def reset_cmd(self, timeout=100000):
        '''
        timeout : uinit ms
        '''
        self.req.close()
        self.req = self.ctx.socket(zmq.REQ)
        self.req.setsockopt(zmq.RCVTIMEO, timeout)
        self.req.connect(self.cmd_port)

    def reset_data(self, timeout=100000):
        '''
        timeout : unit is ms, default 100 second
        '''
        self.sub.close()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.RCVTIMEO, timeout)
        self.sub.connect(self.data_port)
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')


class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods

    send_zipped_pickle is just like send_pyobj, but uses
    zlib to compress the stream before sending.

    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    """

    def send_zipped_pickle(self, obj, flags=0, protocol=-1):
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        # print('zipped pickle is %i bytes' % len(zobj))
        return self.send(zobj, flags=flags)

    def recv_zipped_pickle(self, flags=0):
        """reconstruct a Python object sent with zipped_pickle"""
        zobj = self.recv(flags)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)

    def send_pickle(self, obj, flags=0, protocol=-1):
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        return self.send(pobj, flags=flags)

    def recv_pickle(self, flags=0):
        """reconstruct a Python object sent with zipped_pickle"""
        pobj = self.recv(flags)
        return pickle.loads(pobj)

    def send_array(self, A, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):
        """recv a numpy array"""
        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        A = numpy.frombuffer(msg, dtype=md['dtype'])
        return A.reshape(md['shape'])


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket


if __name__ == '__main__':
    TaskSocket()
