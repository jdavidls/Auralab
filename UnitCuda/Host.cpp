#include "Host.h"
#include "Unit.h"


#define CU_SETUP_ZERO(VAR)											\
	memset(&self->setup.VAR, 0, sizeof(self->setup.VAR));

#define CU_SETUP_DTOH(VAR)											\
	CU_CALL( cuMemcpyDtoH(											\
		&self->setup.VAR,											\
		self->setup_ptr + offsetof(HostSetup, VAR),					\
		sizeof(self->setup.VAR)) );		

#define CU_SETUP_HTOD(VAR)											\
	CU_CALL( cuMemcpyHtoD(											\
		self->setup_ptr + offsetof(HostSetup, VAR),					\
		&self->setup.VAR,											\
		sizeof(self->setup.VAR)) );		

#define CU_SETUP_DTOH_ASYNC(VAR, STREAM)							\
	CU_CALL( cuMemcpyDtoHAsync(										\
		&self->setup.VAR,											\
		self->setup_ptr + offsetof(HostSetup, VAR),					\
		sizeof(self->setup.VAR), STREAM) );		

#define CU_SETUP_HTOD_ASYNC(VAR, STREAM)							\
	CU_CALL( cuMemcpyHtoDAsync(										\
		self->setup_ptr + offsetof(HostSetup, VAR),					\
		&self->setup.VAR,											\
		sizeof(self->setup.VAR), STREAM) );

#define CU_LAUNCH_UNIT_KERNEL(KERNEL)								\
DEQUE_FOREACH(self, chain, unit)									\
	CU_CALL( cuLaunchKernel(										\
		unit->KERNEL,												\
		1, 1, 1, 1, 1, 1,											\
		self->device->prop.sharedMemPerBlock,						\
		0, 0, 0) );

unsigned long upper_power_of_two(unsigned long v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

char* Host_doc = "";

/***************************************************************************************************
*
***************************************************************************************************/

int Host_init(Host* self, PyObject* args, PyObject* kwds)
{
	CU_BEGIN( Host_init );

    static char *kwlist[] = {"device_id", "synchronous", NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwds, "k|b", kwlist,
			&self->device_id,
			&self->synchronous) )
        return -1; 

	//	choose onprocess capsule
	if( self->synchronous )
		self->onProcess = PyCapsule_New(&Host_onProcess_synchronous, "AudioMultibufferProcess", NULL);
	else
		self->onProcess = PyCapsule_New(&Host_onProcess_asynchronous, "AudioMultibufferProcess", NULL);

	//	set computation device
	CU_ASSERT( self->device_id < cuda_device_count, -1, "Invalid device index");
	self->device = cuda_device + self->device_id;

	//	create context & streams
	CU_CALL( cuCtxCreate(&self->cuda_context, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, self->device->handle) );
	CU_CALL( cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 65536) );
	CU_CALL( cuStreamCreate(&self->process_stream, 0) );
	CU_CALL( cuEventCreate(&self->output_ready, CU_EVENT_DISABLE_TIMING) );
	CU_CALL( cuEventCreate(&self->process_finish, CU_EVENT_DISABLE_TIMING) );

	//	send HostSetup to device
	CU_CALL( cuMemAlloc(&self->setup_ptr, sizeof(HostSetup)) );
	CU_CALL( cuMemcpyHtoD(self->setup_ptr, &self->setup, sizeof(HostSetup)) );

	// Detach Cuda context of this thread
	CU_CALL( cuCtxPopCurrent(0) );

	return 0;

	CU_END( -1, cuCtxDestroy(self->cuda_context) );
}

/***************************************************************************************************
*
***************************************************************************************************/

void Host_dealloc(Host* self)
{
	CU_BEGIN( Host_dealloc );

	DEQUE_FOREACH(self, chain, unit)
		Unit_unlink(unit);

	CU_CALL( cuMemFree(self->setup_ptr) );
	CU_CALL( cuCtxDestroy(self->cuda_context) );

	CU_END( (void)0 );
}
 
/***************************************************************************************************
*
***************************************************************************************************/

char* Host_onConnect_doc = "";

PyObject* Host_onConnect(Host* self, PyObject* args, PyObject* kwds)
{
    static char *kwlist[] = {"driver", NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &self->driver) )
        return NULL;

	return Py_None;
}


/***************************************************************************************************
*
***************************************************************************************************/

char* Host_onDisconnect_doc = "";

PyObject* Host_onDisconnect(Host* self)
{
	return Py_None;
}
/***************************************************************************************************
*
***************************************************************************************************/

char* Host_addUnit_doc = "";

PyObject* Host_addUnit(Host* self, PyObject* args, PyObject* kwds)
{
	Unit* unit;

	static char *kwlist[] = {"unit", NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &unit) )
        return NULL;
	
	PY_ASSERT( !self->has_buffers, NULL, " destroy buffers before adding units to the host" );
	PY_ASSERT( PyObject_TypeCheck(unit, &Unit_type), NULL, " argument 1 not is unit type" );
	PY_ASSERT( unit->host == self, NULL, " unit host not is self" );

	if( DEQUE_NOT_LINKED_IN(self, chain, unit) )
	{
		Py_INCREF(unit);
		DEQUE_TAIL_PUSH(self, chain, unit);
	}

	return (PyObject*)unit;
}


/***************************************************************************************************
*
***************************************************************************************************/

char* Host_onCreateBuffers_doc = "";

PyObject* Host_onCreateBuffers(Host* self)
{
	CU_BEGIN( Host_onCreateBuffers );

	PyObject *py_obj;

	//	gets sample rate
	if( (py_obj = PyObject_GetAttrString(self->driver, "sample_rate")) == 0 )
		return NULL;

	self->setup.sample_rate = PyLong_AsUnsignedLong(PyNumber_Long(py_obj));

	//	gets chunk length
	if( !(py_obj = PyObject_GetAttrString(self->driver, "buffer_length")) )
		return NULL;

	self->setup.chunk_length = PyLong_AsUnsignedLong(PyNumber_Long(py_obj));
//	self->setup.chunk_bytes = self->setup.chunk_length * 4; 

	//	gets output count
	if( !(py_obj = PyObject_GetAttrString(self->driver, "output_buffer_count")) )
		return NULL;
	
	self->setup.io.outputs = PyLong_AsUnsignedLong(PyNumber_Long(py_obj));

	//	gets input count
	if( !(py_obj = PyObject_GetAttrString(self->driver, "input_buffer_count")) )
		return NULL;

	self->setup.io.inputs = PyLong_AsUnsignedLong(PyNumber_Long(py_obj));
	self->setup.io.count = self->setup.io.outputs + self->setup.io.inputs;

	//	setup io buffers

	unsigned long idx = 0;

	for(unsigned long i = 0; i < self->setup.io.outputs; i++, idx++)
	{
		HostBuffer* buffer = self->setup.io.buffer + idx;
		buffer->format = HOST_BUFFER_FORMAT_INT32;
		buffer->length = self->setup.chunk_length;
	}

	for(unsigned long i = 0; i < self->setup.io.inputs; i++, idx++)
	{
		HostBuffer* buffer = self->setup.io.buffer + idx;
		buffer->format = HOST_BUFFER_FORMAT_INT32;
		buffer->length = self->setup.chunk_length;
	}

	//	enter cuda context
	CU_CALL( cuCtxPushCurrent(self->cuda_context) );

	// upload setup
	CU_CALL( cuMemcpyHtoD(self->setup_ptr, &self->setup, sizeof(HostSetup)) );
	
	// launch onreateBuffers kernel
	CU_LAUNCH_UNIT_KERNEL( onCreateBuffers );
	
	// read aux bufferss setup
	CU_SETUP_DTOH( io );

	//	prepare all buffers
	for(unsigned long i = 0; i < self->setup.io.count; i++)
	{
		HostBuffer* buffer = self->setup.io.buffer + i;
		buffer->length = upper_power_of_two(buffer->length);
		buffer->mask = buffer->length - 1;
		buffer->bytes = buffer->length * HOST_BUFFER_FORMAT_BYTES(buffer->format);
		buffer->offset = 0;

		if( buffer->bytes )
			CU_CALL( cuMemAlloc(&buffer->ptr, buffer->bytes) );

		printf("  cuda alloc buffer %d: %p\n", i, buffer->ptr);
	}

	self->has_buffers = true;

	DEQUE_FOREACH(self, chain, unit)
		Unit_launchQueue(unit);

	// finish
	CU_CALL( cuCtxPopCurrent(0) );

	return Py_None;
	
	CU_END( NULL, cuCtxPopCurrent(0) );
}

/***************************************************************************************************
*
***************************************************************************************************/

char* Host_onDestroyBuffers_doc = "";

PyObject* Host_onDestroyBuffers(Host* self)
{
	CU_BEGIN( Host_onDestroyBuffers );
	
	CU_CALL( cuCtxPushCurrent(self->cuda_context) )

	//	launch unit.onDestroyBuffers kernel
	CU_LAUNCH_UNIT_KERNEL( onDestroyBuffers );

	// realeases input buffers
	for(unsigned long i = 0; i < self->setup.io.count; i++)
	{
		HostBuffer *buffer = self->setup.io.buffer + i;

		CU_CALL( cuMemFree(buffer->ptr) );

		buffer->format = 0;
		buffer->mask = 0;
		buffer->length = 0;
		buffer->bytes = 0;
		buffer->ptr = 0;
	}

	self->setup.io.outputs = 0;
	self->setup.io.inputs = 0;
	self->setup.io.count = 0;

	CU_SETUP_HTOD( io );

	DEQUE_FOREACH(self, chain, unit)
		Unit_launchQueue(unit);

	self->has_buffers = false;

	//	pop context
	CU_CALL( cuCtxPopCurrent(0) );

	return Py_None;
	
	CU_END( NULL, cuCtxPopCurrent(0) );
}

/***************************************************************************************************
*
***************************************************************************************************/

char* Host_onPlay_doc = "";

PyObject* Host_onPlay(Host* self)
{
	CU_BEGIN( Host_onPlay )

	CU_CALL( cuCtxPushCurrent(self->cuda_context) )

	//	reset input buffers
	for(unsigned long i = 0; i < self->setup.io.count; i++)
	{
		HostBuffer *buffer = self->setup.io.buffer + i;
	
		CU_CALL( cuMemsetD8(buffer->ptr, 0, buffer->bytes) );
	
		buffer->offset = 0;
	}

	CU_SETUP_HTOD( io );

	//	call to unit.onPlay
	CU_LAUNCH_UNIT_KERNEL( onPlay );

	DEQUE_FOREACH(self, chain, unit)
	{
		Unit_launchQueue(unit);
		Unit_prepareProcess(unit);
	}

	CU_CALL( cuCtxPopCurrent(0) );

	return Py_None;
	
	CU_END( NULL, cuCtxPopCurrent(0) )
}

/***************************************************************************************************
*
***************************************************************************************************/

char* Host_onStop_doc = "";

PyObject* Host_onStop(Host* self)
{
	CU_BEGIN( Host_onStop );

	CU_CALL( cuCtxPushCurrent(self->cuda_context) );

	//	call to unit.onStop
	CU_LAUNCH_UNIT_KERNEL( onStop );

	DEQUE_FOREACH(self, chain, unit)
		Unit_launchQueue(unit);

	CU_CALL( cuCtxPopCurrent(0) );

	return Py_None;
	
	CU_END( NULL, cuCtxPopCurrent(0) );
}

/***************************************************************************************************
*
***************************************************************************************************/
#define INT_TO_FLOAT_MUL	(1.L / 0x7fffffffL)
#define FLOAT_TO_INT_MUL	(0x7fffffffL)

#define INT_TO_FLOAT(INT)	float( INT * INT_TO_FLOAT_MUL )
#define FLOAT_TO_INT(FLOAT) int( FLOAT * FLOAT_TO_INT_MUL )

#define PI2	3.1415


char* Host_onProcess_doc = "";

void Host_onProcess_asynchronous(Host* self, void** input, void** output, void(*outputsReady)())
{
	CU_BEGIN( Host_onProcess_asynchronous );
	
	CU_CALL( cuCtxPushCurrent(self->cuda_context) );
	
	//CU_CALL( cuStreamWaitEvent(self->process_stream, self->process_finish, 0) );

	unsigned long idx = 0;

	//	RECV output buffers from device
	for(unsigned long i = 0; i < self->setup.io.outputs; i++, idx++)
	{
		HostBuffer* buffer = self->setup.io.buffer + idx;
		
		unsigned long format_bytes = HOST_BUFFER_FORMAT_BYTES(buffer->format);


		CU_CALL( cuMemcpyDtoHAsync( 
			output[i],
			buffer->ptr + buffer->offset * format_bytes,
			self->setup.chunk_length * format_bytes,
			self->process_stream) );
		
		buffer->offset += self->setup.chunk_length;
		buffer->offset &= buffer->mask;
	}
	
	CU_CALL( cuEventRecord(self->output_ready, self->process_stream) );
	
	//	SEND input buffers to device
	for(unsigned long i = 0; i < self->setup.io.inputs; i++, idx++)
	{
		HostBuffer* buffer = self->setup.io.buffer + idx;

		unsigned long format_bytes = HOST_BUFFER_FORMAT_BYTES(buffer->format);

		CU_CALL( cuMemcpyHtoDAsync(
			buffer->ptr + buffer->offset * format_bytes,
			input[i], 
			self->setup.chunk_length * format_bytes,
			self->process_stream) );

		buffer->offset += self->setup.chunk_length;
		buffer->offset &= buffer->mask;
	}

	// update buffers info
	for(;idx < self->setup.io.count; idx++)
	{
		HostBuffer* buffer = self->setup.io.buffer + idx;
		buffer->offset += self->setup.chunk_length;
		buffer->offset &= buffer->mask;
	}

	CU_SETUP_HTOD_ASYNC( io, self->process_stream );

	//	Launch process queue
	DEQUE_FOREACH(self, chain, unit)
		Unit_launchProcess(unit);

	//	wait until outputs ready
	CU_CALL( cuStreamWaitEvent(self->process_stream, self->output_ready, 0) );

	outputsReady();

	//	finish
	CU_CALL( cuCtxPopCurrent(0) );
	
	self->process_count++;
	return;
	CU_END( (void)0, printf("processError\n"), cuCtxPopCurrent(0) )
}


/***************************************************************************************************
*
***************************************************************************************************/



void Host_onProcess_synchronous(Host* self, int** inputs, int** outputs, void(*outputsReady)())
{
	self->process_count++;
	
	float freq = 1.f / self->setup.chunk_length;

	for(unsigned long i = 0; i < self->setup.chunk_length; i++)
	{
		inputs[0][i] = 	rand() << 17;
		outputs[0][i] = FLOAT_TO_INT( sin( PI2 * i * freq ) );
	}

	outputsReady();
}
