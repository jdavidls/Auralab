#pragma once

#include "UnitCuda.h"
#include "../utils/list.h"
#include "ids.h"

struct Unit;


struct Host
{
	PyObject_HEAD

	unsigned long device_id;
	char synchronous;
	char has_buffers;

	unsigned long process_count;

	CudaDevice* device;
	CUcontext cuda_context;

	CUstream process_stream;
	CUevent	output_ready;
	CUevent	process_finish;

	PyObject* driver;
	
	HostSetup setup;
	CUdeviceptr setup_ptr;

	PyObject* onProcess;

	Unit* unit_chain;
	
	DEQUE(Unit, chain);
};

extern PyTypeObject Host_type;
extern char* Host_doc;

extern int Host_init(Host*, PyObject*, PyObject*);
extern void Host_dealloc(Host*);

extern char* Host_addUnit_doc;
PyObject* Host_addUnit(Host* self, PyObject*, PyObject*);

extern char* Host_onProcess_doc;
extern void Host_onProcess_synchronous(Host*, int**, int**, void(*)());
extern void Host_onProcess_asynchronous(Host*, void**, void**, void(*)());

extern char* Host_onConnect_doc;
PyObject* Host_onConnect(Host* self, PyObject* args, PyObject* kwds);

extern char* Host_onDisconnect_doc;
PyObject* Host_onDisconnect(Host* self);

extern char* Host_onCreateBuffers_doc;
PyObject* Host_onCreateBuffers(Host* self);

extern char* Host_onDestroyBuffers_doc;
PyObject* Host_onDestroyBuffers(Host* self);

extern char* Host_onPlay_doc;
PyObject* Host_onPlay(Host* self);

extern char* Host_onStop_doc;
PyObject* Host_onStop(Host* self);
