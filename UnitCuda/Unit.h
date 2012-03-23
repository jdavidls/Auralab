#pragma once

#include "UnitCuda.h"
#include "ids.h"

struct Host;

struct Unit
{
	PyObject_HEAD
	
	Host* host;
	char* filename;
	CUmodule module;

	CUdeviceptr setup_ptr;
	size_t setup_bytes;
	UnitSetup setup;



	CUfunction onLoad;
	CUfunction onUnload;
	CUfunction onCreateBuffers;
	CUfunction onDestroyBuffers;
	CUfunction onPlay;
	CUfunction onStop;

	DEQUE_NODE(Unit, chain);
};

extern PyTypeObject Unit_type;
extern PyTypeObject UnitGlobal_type;
extern PyTypeObject UnitKernel_type;

extern char* Unit_doc;
extern int Unit_init(Unit*, PyObject*, PyObject*);
extern void Unit_dealloc(Unit*);
extern PyObject* Unit_unlink(Unit*);
extern void Unit_launchQueue(Unit*);
extern void Unit_prepareProcess(Unit*);
extern void Unit_launchProcess(Unit*);

extern PyObject* Unit_getter(Unit*, PyObject*, PyObject*);
