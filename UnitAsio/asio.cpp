#include <intrin.h>
#include "asio_driver.h"

#define SLOTS								8

#define SAFE( CALL )						\
if( CALL )									\
	goto asioError;

#undef DELETE
#define DELETE( OBJECT )					\
if( OBJECT )								\
{											\
	delete OBJECT;							\
	OBJECT = NULL;							\
}

#define PY_ASSERT(EVAL, MSG, RETVAL)		\
if( !(EVAL) )								\
{											\
	PyErr_SetString(exception, MSG);		\
	return RETVAL;							\
}


//--------------------------------------------------------------------------------------------------
static ASIOCallbacks callbacks[SLOTS] = 
{
	{&cbBufferSwitch<0>, &cbSampleRate<0>, &cbMessage<0>, &cbBufferSwitchTime<0>},
	{&cbBufferSwitch<1>, &cbSampleRate<1>, &cbMessage<1>, &cbBufferSwitchTime<1>},
	{&cbBufferSwitch<2>, &cbSampleRate<2>, &cbMessage<2>, &cbBufferSwitchTime<2>},
	{&cbBufferSwitch<3>, &cbSampleRate<3>, &cbMessage<3>, &cbBufferSwitchTime<3>},
	{&cbBufferSwitch<4>, &cbSampleRate<4>, &cbMessage<4>, &cbBufferSwitchTime<4>},
	{&cbBufferSwitch<5>, &cbSampleRate<5>, &cbMessage<5>, &cbBufferSwitchTime<5>},
	{&cbBufferSwitch<6>, &cbSampleRate<6>, &cbMessage<6>, &cbBufferSwitchTime<6>},
	{&cbBufferSwitch<7>, &cbSampleRate<7>, &cbMessage<7>, &cbBufferSwitchTime<7>}
};

static Driver* driver_slot[SLOTS] = {0,0,0,0,0,0,0,0};
//static long buffer_lock[SLOTS] = {1,1,1,1,1,1,1,1};

static PyObject *exception;

static int Driver_init(Driver *self, PyObject *args, PyObject *kwds);
static void Driver_dealloc(Driver* self);

static PyObject* Driver_createBuffers(Driver *self, PyObject *args, PyObject* kwds);
static PyObject* Driver_destroyBuffers(Driver*);

static PyObject* Driver_play(Driver*);
static PyObject* Driver_stop(Driver*);

static PyObject* Driver_connect(Driver *self, PyObject* args, PyObject *kwds);
static PyObject* Driver_disconnect(Driver *self);

static void Driver_onProcess(Driver*, void**, void**, void(*)());

//--------------------------------------------------------------------------------------------------
static int Driver_init(Driver *self, PyObject *args, PyObject *kwds)
{
	PyUnicodeObject *clsid_unicode;

	self->slot = -1;
	self->onProcess = PyCapsule_New(&Driver_onProcess, "AudioMultibufferProcess", NULL);

    static char *kwlist[] = {"clsid", NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwds, "U", kwlist, &clsid_unicode) )
        return -1; 
	
	wchar_t* clsid = PyUnicode_AsWideCharString((PyObject*)clsid_unicode, NULL);

	IID iid;

	PY_ASSERT( CLSIDFromString(clsid, &iid) == S_OK,
		"clsid to iid conversion error", -1);
	
	PY_ASSERT( CoCreateInstance(iid, 0, CLSCTX_INPROC_SERVER, iid, (LPVOID*)&self->iasio) == S_OK,
		"error instanciating asio com object", -1);

	self->iasio->getDriverName(self->name);
	self->version = self->iasio->getDriverVersion();
	
	PY_ASSERT( self->iasio->getBufferSize(
		&self->min_buffer_length, 
		&self->max_buffer_length, 
		&self->preferred_buffer_length, 
		&self->buffer_granularity) == ASE_OK,
		"error calling asio->getBufferSize", -1);

	PY_ASSERT( self->iasio->getChannels(
		&self->input_channel_count,
		&self->output_channel_count) == ASE_OK,
		"error calling asio->getChannels", -1);

	self->input_channel_info = new ASIOChannelInfo[self->input_channel_count];
	
	for(int i = 0; i < self->input_channel_count; i++)
	{
		self->input_channel_info[i].channel = i;
		self->input_channel_info[i].isInput = true;
		self->input_channel_info[i].isActive = false;

		PY_ASSERT( self->iasio->getChannelInfo(self->input_channel_info + i) == ASE_OK,
			"error calling asio->getChannelInfo", -1);
		
//		printf("input_channel[%d].type = %d\n", i, self->input_channel_info[i].type);
	}

	self->output_channel_info = new ASIOChannelInfo[self->output_channel_count];

	for(int i = 0; i < self->output_channel_count; i++)
	{
		self->output_channel_info[i].channel = i;
		self->output_channel_info[i].isInput = false;
		self->output_channel_info[i].isActive = false;

		PY_ASSERT( self->iasio->getChannelInfo(self->output_channel_info + i) == ASE_OK,
			"error calling asio->getChannelInfo", -1);
		
//		printf("output_channel[%d].type = %d\n", i, self->output_channel_info[i].type);
	}

	//	autoconnection
	//	PyObject_CallMethod((PyObject*)self, "connect", "O", self);

    return 0;
}

//--------------------------------------------------------------------------------------------------
static void Driver_dealloc(Driver* self)
{
	Driver_disconnect(self);

	DELETE( self->input_channel_info );
	DELETE( self->output_channel_info );
	
	if( self->iasio )
	{
		self->iasio->Release();
		self->iasio = 0;
	}

	Py_TYPE(self)->tp_free((PyObject*)self);
}

//----------------------------------------------------------------------------------
PyObject* Driver_connect(Driver *self, PyObject* args, PyObject *kwds)
{
	static char *kwlist[] = {"host", NULL};

	PyObject *host, *onProcess;

    if( !PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &host) )
        return NULL; 

	onProcess = PyObject_GetAttrString(host, "onProcess");

	PY_ASSERT(onProcess && PyCapsule_CheckExact(onProcess), 
		"Host object haven't an onProcess capsule", NULL);

	OnProcess *host_onProcess = (OnProcess*)
		PyCapsule_GetPointer(onProcess, "AudioMultibufferProcess");

	if(host_onProcess == NULL)
		return NULL;

	Driver_disconnect(self);

	Py_IncRef(host);
	self->host = host;
	self->host_onProcess = host_onProcess;

	return PyObject_CallMethod(self->host, "onConnect", "O", self);
}

//----------------------------------------------------------------------------------
PyObject* Driver_disconnect(Driver *self)
{
	Driver_destroyBuffers(self);
	
	PyObject* result = Py_None;

	if( self->host )
	{
		result = PyObject_CallMethod(self->host, "onDisconnect", "");
		Py_DecRef(self->host);
	}

	self->host = NULL;
	self->onProcess = NULL;

	return result;
}

//--------------------------------------------------------------------------------------------------
static PyObject * Driver_createBuffers(Driver *self, PyObject *args, PyObject* kwds)
{
	PyObject *inputs, *outputs;

	long buffer_length = -1;

	static char *kwlist[] = {"inputs", "outputs", "buffer_length", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO|k", kwlist, &inputs, &outputs, &buffer_length) )
        return NULL; 

	PY_ASSERT(PyObject_TypeCheck(inputs, &PyTuple_Type),
		"asio.Driver.createBuffers: inputs must be a tuple of input channel indexes", NULL);

	PY_ASSERT(PyObject_TypeCheck(outputs, &PyTuple_Type),
		"asio.Driver.createBuffers: outputs must be a tuple of input channel indexes", NULL);

	PY_ASSERT(self->host, "asio.Driver.createBuffers: Driver is disconnected", NULL);

	Driver_destroyBuffers(self);

	//	assign an slot
	for(long slot = 0; true; slot++)
	{
		PY_ASSERT(slot < SLOTS, "asio.Driver.createBuffers: not found an available slot", NULL);

		if( driver_slot[slot] == NULL )
		{
			driver_slot[slot] = self;
			self->slot = slot;
			break;
		}
	}

	self->buffer_length = buffer_length == -1? self->preferred_buffer_length: buffer_length;
	
	self->input_buffer_count = PyObject_Length(inputs);
	self->output_buffer_count = PyObject_Length(outputs);

	self->buffer_count = self->input_buffer_count + self->output_buffer_count;
	self->buffer_info = new ASIOBufferInfo[self->buffer_count];
	
	self->input_buffer[0] = new void*[self->input_buffer_count];
	self->input_buffer[1] = new void*[self->input_buffer_count];
	self->output_buffer[0] = new void*[self->output_buffer_count];
	self->output_buffer[1] = new void*[self->output_buffer_count];

	self->input_buffer_info = self->buffer_info;
	self->output_buffer_info = self->buffer_info + self->input_buffer_count;

	for(long idx = 0; idx < self->input_buffer_count; idx++)
	{
		self->input_buffer_info[idx].buffers[0] = 0;
		self->input_buffer_info[idx].buffers[1] = 0;
		self->input_buffer_info[idx].channelNum = PyLong_AsLong(PyTuple_GetItem(inputs, idx));
		self->input_buffer_info[idx].isInput = true;
	}

	for(long idx = 0; idx < self->output_buffer_count; idx++)
	{
		self->output_buffer_info[idx].buffers[0] = 0;
		self->output_buffer_info[idx].buffers[1] = 0;
		self->output_buffer_info[idx].channelNum = PyLong_AsLong(PyTuple_GetItem(outputs, idx));
		self->output_buffer_info[idx].isInput = false;
	}

	if( self->iasio->createBuffers(self->buffer_info, self->buffer_count, self->buffer_length, callbacks + self->slot) )
		goto asioError;

	if( self->iasio->getLatencies(&self->input_latency, &self->output_latency) )
		goto asioError;

	if( self->iasio->getSampleRate(&self->sample_rate) )
		goto asioError;

	for(long i = 0; i < self->output_buffer_count; i++)
	{
		self->output_buffer[0][i] = self->output_buffer_info[i].buffers[0];
		self->output_buffer[1][i] = self->output_buffer_info[i].buffers[1];

		printf("output buffer %d A:%p B:%p\n", i, self->output_buffer[0][i], self->output_buffer[1][i]);
	}

	for(long i = 0; i < self->input_buffer_count; i++)
	{
		self->input_buffer[0][i] = self->input_buffer_info[i].buffers[0];
		self->input_buffer[1][i] = self->input_buffer_info[i].buffers[1];
	}

	return PyObject_CallMethod(self->host, "onCreateBuffers", "");

asioError:

	//	release slot
	driver_slot[self->slot] = 0;
	self->slot = -1;

	DELETE(self->buffer_info);
	DELETE(self->input_buffer[0]);
	DELETE(self->input_buffer[1]);
	DELETE(self->output_buffer[0]);
	DELETE(self->output_buffer[1]);

	self->buffer_length = 0;
	self->buffer_count = 0;
	self->input_buffer_count = 0;
	self->output_buffer_count = 0;

	char error_message[256];
	self->iasio->getErrorMessage(error_message);
	PyErr_SetString(exception, error_message);

	return NULL;
}

//----------------------------------------------------------------------------------
PyObject* Driver_destroyBuffers(Driver* self)
{
	Driver_stop(self);

	if(self->slot != -1)
	{
		driver_slot[self->slot] = 0;
		self->slot = -1;

		DELETE(self->buffer_info);
		DELETE(self->input_buffer[0]);
		DELETE(self->input_buffer[1]);
		DELETE(self->output_buffer[0]);
		DELETE(self->output_buffer[1]);

		self->buffer_length = 0;
		self->buffer_count = 0;
		self->input_buffer_count = 0;
		self->output_buffer_count = 0;

		if( self->iasio->disposeBuffers() )
			goto asioError;

		if( self->host )
			return PyObject_CallMethod(self->host, "onDestroyBuffers", "");
	}
	
	return Py_None;

asioError:
	char error_message[256];
	self->iasio->getErrorMessage(error_message);
	PyErr_SetString(exception, error_message);

	return NULL;
}

//----------------------------------------------------------------------------------
PyObject* Driver_play(Driver* self)
{
	PY_ASSERT( self->slot >= 0, "asio.Driver.play: slot not assigned", NULL );
	PY_ASSERT( !self->is_playing, "asio.Driver.play: already playing", NULL );


	if( self->iasio->start() )
		goto asioError;
	
	PyObject* result = self->host? PyObject_CallMethod(self->host, "onPlay", ""): Py_None;

	self->is_playing = true;

	return result;

asioError:
	char error_message[256];
	self->iasio->getErrorMessage(error_message);
	PyErr_SetString(exception, error_message);

	return NULL;
}

//----------------------------------------------------------------------------------
PyObject* Driver_stop(Driver* self)
{
	if( self->slot < 0 )
		return Py_None;

	if( !self->is_playing )
		return Py_None;

	if( self->iasio->stop() )
		goto asioError;

	// TODO: wait until buffer_refcount > 0

	self->is_playing = false;

	if(	self->host	)
		return PyObject_CallMethod(self->host, "onStop", NULL);
	else
		return Py_None;

asioError:
	char error_message[256];
	self->iasio->getErrorMessage(error_message);
	PyErr_SetString(exception, error_message);

	return NULL;
}

//----------------------------------------------------------------------------------
PyObject* Driver_controlPannel(Driver* self)
{
	if( self->iasio->controlPanel() != ASE_OK )
		goto asioError;

	return Py_None;

asioError:
	char error_message[256];
	self->iasio->getErrorMessage(error_message);
	PyErr_SetString(exception, error_message);

	return NULL;
}

//----------------------------------------------------------------------------------
PyObject* Driver_inputChannelName(Driver* self, PyObject* args, PyObject* kwds)
{
	long channel_idx;

	static char *kwlist[] = {"channel_idx", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &channel_idx) )
        return NULL; 

	if( channel_idx > self->input_channel_count )
	{
		PyErr_SetString(PyExc_IndexError, "channel out of index");
		return NULL;
	}

	return PyUnicode_FromString(self->input_channel_info[channel_idx].name);
}

//----------------------------------------------------------------------------------
PyObject* Driver_outputChannelName(Driver* self, PyObject* args, PyObject* kwds)
{
	long channel_idx;

	static char *kwlist[] = {"channel_idx", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &channel_idx) )
        return NULL; 

	if( channel_idx > self->output_channel_count )
	{
		PyErr_SetString(PyExc_IndexError, "channel out of index");
		return NULL;
	}

	return PyUnicode_FromString(self->output_channel_info[channel_idx].name);
}

//----------------------------------------------------------------------------------
PyObject* Driver_inputChannelFormat(Driver* self, PyObject* args, PyObject* kwds)
{
	long channel_idx;

	static char *kwlist[] = {"channel_idx", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &channel_idx) )
        return NULL; 

	if( channel_idx > self->input_channel_count )
	{
		PyErr_SetString(PyExc_IndexError, "channel out of index");
		return NULL;
	}

	return PyLong_FromLong(self->input_channel_info[channel_idx].type);
}

//----------------------------------------------------------------------------------
PyObject* Driver_outputChannelFormat(Driver* self, PyObject* args, PyObject* kwds)
{
	long channel_idx;

	static char *kwlist[] = {"channel_idx", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &channel_idx) )
        return NULL; 

	if( channel_idx > self->output_channel_count )
	{
		PyErr_SetString(PyExc_IndexError, "channel out of index");
		return NULL;
	}

	return PyLong_FromLong(self->output_channel_info[channel_idx].type);
}

//----------------------------------------------------------------------------------
PyObject* Driver_onConnect(Driver* self, PyObject* args, PyObject* kwds)
{
	printf("onConnect\n");
	return Py_None;
}

//----------------------------------------------------------------------------------
PyObject* Driver_onDisconnect(Driver* self)
{
	printf("onDisconnect\n");
	return Py_None;
}

//----------------------------------------------------------------------------------
PyObject* Driver_onCreateBuffers(Driver* self, PyObject* args, PyObject* kwds)
{
	printf("onCreateBuffers\n");
	return Py_None;
}

//----------------------------------------------------------------------------------
PyObject* Driver_onDestroyBuffers(Driver* self, PyObject* args, PyObject* kwds)
{
	printf("onDestroyBuffers\n");
	return Py_None;
}

//----------------------------------------------------------------------------------
PyObject* Driver_onPlay(Driver* self)
{
	printf("onPlay\n");
	return Py_None;
}

//----------------------------------------------------------------------------------
PyObject* Driver_onStop(Driver* self)
{
	printf("onStop\n");
	return Py_None;
}

//----------------------------------------------------------------------------------
void Driver_onProcess(Driver* self, void** inputs, void** outputs, void(*outputReady)() )
{
	//printf("onProcess");

	if(outputReady)	outputReady();
}

//----------------------------------------------------------------------------------
template <unsigned SLOT>
void cbOutputReady()
{
	Driver* self = driver_slot[SLOT];

	// convierte el outputbufer a floats..

	if( self && self->is_playing )
		self->iasio->outputReady();
}

//----------------------------------------------------------------------------------
template <unsigned SLOT>
void cbBufferSwitch(long index, ASIOBool process_now)
{
	Driver* self = driver_slot[SLOT];
	
	if( self && self->is_playing )
	{
		//for(long i =0 ; i < self->buffer_length; i++)
//			((int**)self->output_buffer[index])[0][i] = rand()<<16;

		// entra en el buffer
		// convierte el input buffer a floats..
		self->host_onProcess(self->host, self->input_buffer[index], self->output_buffer[index], &cbOutputReady<SLOT>);
		// sale del buffer
	}
}

//----------------------------------------------------------------------------------
template <unsigned SLOT>
ASIOTime *cbBufferSwitchTime(ASIOTime *timeInfo, long index, ASIOBool process_now)
{
	callbacks[SLOT].bufferSwitch(index, process_now);
	return NULL;
}

//----------------------------------------------------------------------------------
template <unsigned SLOT>
void cbSampleRate(ASIOSampleRate sample_rate)
{
	// do whatever you need to do if the sample rate changed
	// usually this only happens during external sync.
	// Audio processing is not stopped by the driver, actual sample rate
	// might not have even changed, maybe only the sample rate status of an
	// AES/EBU or S/PDIF digital input at the audio device.
	// You might have to update time/sample related conversion routines, etc.
	Driver* drv = driver_slot[SLOT];

	if( drv )
	{
		/*
		if(drv->_sample_rate != sample_rate)
		{
			drv->_sample_rate = sample_rate;
			drv->_host->onSampleRateChanged(sample_rate);
		}*/
	}
}

//----------------------------------------------------------------------------------
template <unsigned SLOT>
long cbMessage(long selector, long value, void* message, double* opt)
{
	Driver* drv = driver_slot[SLOT];

	if( drv == 0 )
		return 0;

	long ret = 0;
	switch(selector)
	{
		case kAsioSelectorSupported:
			if(value == kAsioResetRequest
			|| value == kAsioEngineVersion
			|| value == kAsioResyncRequest
			|| value == kAsioLatenciesChanged
			// the following three were added for ASIO 2.0, you don't necessarily have to support them
			|| value == kAsioSupportsTimeInfo
			|| value == kAsioSupportsTimeCode
			|| value == kAsioSupportsInputMonitor)
				ret = 1L;
			break;

		case kAsioResetRequest:
			// defer the task and perform the reset of the driver during the next "safe" situation
			// You cannot reset the driver right now, as this code is called from the driver.
			// Reset the driver is done by completely destruct is. I.e. ASIOStop(), ASIODisposeBuffers(), Destruction
			// Afterwards you initialize the driver again.
			
			//if( drv->_host )		drv->_host->onResetRequest();
			return 1L;

		case kAsioResyncRequest:
			// This informs the application, that the driver encountered some non fatal data loss.
			// It is used for synchronization purposes of different media.
			// Added mainly to work around the Win16Mutex problems in Windows 95/98 with the
			// Windows Multimedia system, which could loose data because the Mutex was hold too long
			// by another thread.
			// However a driver can issue it in other situations, too.
			
			//if( drv->_host )		drv->_host->onResyncRequest();
			return 1L;

		case kAsioLatenciesChanged:
			// This will inform the host application that the drivers were latencies changed.
			// Beware, it this does not mean that the buffer sizes have changed!
			// You might need to update internal delay data.
			
			//drv->updateLatencies();
			
			return 1L;

		case kAsioEngineVersion:
			// return the supported ASIO version of the host application
			// If a host applications does not implement this selector, ASIO 1.0 is assumed
			// by the driver
			return 2L;

		case kAsioSupportsTimeInfo:
			// informs the driver wether the asioCallbacks.bufferSwitchTimeInfo() callback
			// is supported.
			// For compatibility with ASIO 1.0 drivers the host application should always support
			// the "old" bufferSwitch method, too.
			return 1L;

		case kAsioSupportsTimeCode:
			// informs the driver wether application is interested in time code info.
			// If an application does not need to know about time code, the driver has less work
			// to do.
			return 0L;//drv->_support_time_code;
	}

	return ret;
}


//--------------------------------------------------------------------------------------------------
static PyMemberDef Driver_members[] = {
	{"name", T_STRING_INPLACE, offsetof(Driver, name), READONLY, "Driver name"},
	{"version", T_LONG, offsetof(Driver, version), READONLY, "Driver version"},
	{"input_channel_count", T_LONG, offsetof(Driver, input_channel_count), READONLY, ""},
	{"output_channel_count", T_LONG, offsetof(Driver, output_channel_count), READONLY, ""},
	{"sample_rate", T_DOUBLE, offsetof(Driver, sample_rate), READONLY, "Sample Rate"},
	{"input_latency", T_LONG, offsetof(Driver, input_latency), READONLY, "Input latency"},
	{"output_latency", T_LONG, offsetof(Driver, output_latency), READONLY, "Output latency"},
	{"min_buffer_length", T_LONG, offsetof(Driver, min_buffer_length), READONLY, ""},
	{"max_buffer_length", T_LONG, offsetof(Driver, max_buffer_length), READONLY, ""},
	{"preferred_buffer_length", T_LONG, offsetof(Driver, preferred_buffer_length), READONLY, ""},
	{"buffer_granularity", T_LONG, offsetof(Driver, buffer_granularity), READONLY, ""},
	{"buffer_length", T_LONG, offsetof(Driver, buffer_length), READONLY, ""},
	{"input_buffer_count", T_LONG, offsetof(Driver, input_buffer_count), READONLY, ""},
	{"output_buffer_count", T_LONG, offsetof(Driver, output_buffer_count), READONLY, ""},
	{"is_playing", T_BOOL, offsetof(Driver, is_playing), READONLY, ""},
	{"host", T_OBJECT, offsetof(Driver, host), READONLY, ""},
	{"onProcess", T_OBJECT, offsetof(Driver, onProcess), READONLY, ""},
    {NULL}  /* Sentinel */
};

static PyGetSetDef Driver_getseters[] = {
//    {"first", (getter)Noddy_getfirst, (setter)Noddy_setfirst, "first name", NULL},
//    {"last", (getter)Noddy_getlast, (setter)Noddy_setlast, "last name", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Driver_methods[] = {
	{"connect", (PyCFunction)Driver_connect, METH_KEYWORDS | METH_VARARGS, ""},
	{"disconnect", (PyCFunction)Driver_disconnect, METH_NOARGS, ""},
	{"createBuffers", (PyCFunction)Driver_createBuffers, METH_KEYWORDS | METH_VARARGS, ""},
	{"destroyBuffers", (PyCFunction)Driver_destroyBuffers, METH_NOARGS, ""},
	{"controlPannel", (PyCFunction)Driver_controlPannel, METH_NOARGS, ""},
	{"inputChannelName", (PyCFunction)Driver_inputChannelName, METH_KEYWORDS | METH_VARARGS, ""},
	{"outputChannelName", (PyCFunction)Driver_outputChannelName, METH_NOARGS, ""},
	{"inputChannelFormat", (PyCFunction)Driver_inputChannelFormat, METH_KEYWORDS | METH_VARARGS, ""},
	{"outputChannelFormat", (PyCFunction)Driver_outputChannelFormat, METH_NOARGS, ""},
	{"play", (PyCFunction)Driver_play, METH_NOARGS, ""},
	{"stop", (PyCFunction)Driver_stop, METH_NOARGS, ""},
	{"onConnect", (PyCFunction)Driver_onConnect, METH_KEYWORDS | METH_VARARGS, ""},
	{"onDisconnect", (PyCFunction)Driver_onDisconnect, METH_NOARGS, ""},
	{"onCreateBuffers", (PyCFunction)Driver_onCreateBuffers, METH_KEYWORDS | METH_VARARGS, ""},
	{"onDestroyBuffers", (PyCFunction)Driver_onDestroyBuffers, METH_NOARGS, ""},
	{"onPlay", (PyCFunction)Driver_onPlay, METH_NOARGS, ""},
	{"onStop", (PyCFunction)Driver_onStop, METH_NOARGS, ""},
	{NULL}  /* Sentinel */
};

static PyTypeObject asio_driver_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "unit._asio.Driver",         /* tp_name */
    sizeof(Driver),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)Driver_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "Asio Driver",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Driver_methods,             /* tp_methods */
    Driver_members,             /* tp_members */
    Driver_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Driver_init,      /* tp_init */
    0,                         /* tp_alloc */
    0//	Noddy_new,                 /* tp_new */
};

static PyModuleDef asio_module = {
    PyModuleDef_HEAD_INIT,
    "unit._asio",
    "",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

//--------------------------------------------------------------------------------------------------
PyMODINIT_FUNC PyInit__asio(void) 
{
	CoInitialize(0);

    PyObject* m;

    asio_driver_type.tp_new = PyType_GenericNew;

	if (PyType_Ready(&asio_driver_type) < 0)
        return NULL;

	exception = PyErr_NewException("unit._asio.AsioException", NULL, NULL);

    m = PyModule_Create(&asio_module);
    if (m == NULL)
        return NULL;

	Py_INCREF(exception);
	PyModule_AddObject(m, "AsioException", exception);

    Py_INCREF(&asio_driver_type);
	PyModule_AddObject(m, "Driver", (PyObject *)&asio_driver_type);

	return m;
}
