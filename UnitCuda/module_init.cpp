
#include "UnitCuda.h"

#include "Host.h"
#include "Unit.h"

PyObject* cuda_exception = NULL;

unsigned long cuda_version = 0;

unsigned long cuda_device_count = 0;

CudaDevice* cuda_device = NULL;

/***************************************************************************************************
*
***************************************************************************************************/

static PyMemberDef Host_members[] = {
	{"onProcess", T_OBJECT, offsetof(Host, onProcess), READONLY, Host_onProcess_doc}, 
	{"process_count", T_ULONG, offsetof(Host, process_count), READONLY, "onProcess call counter"},
    {NULL}  /* Sentinel */
};

static PyGetSetDef Host_getseters[] = {
//    {"first", (getter)Noddy_getfirst, (setter)Noddy_setfirst, "first name", NULL},
//    {"last", (getter)Noddy_getlast, (setter)Noddy_setlast, "last name", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Host_methods[] = {
	{"addUnit", (PyCFunction)Host_addUnit, METH_KEYWORDS | METH_VARARGS, Host_addUnit_doc},
	{"onConnect", (PyCFunction)Host_onConnect, METH_KEYWORDS | METH_VARARGS, Host_onConnect_doc},
	{"onDisconnect", (PyCFunction)Host_onDisconnect, METH_NOARGS, Host_onDisconnect_doc},
	{"onCreateBuffers", (PyCFunction)Host_onCreateBuffers, METH_KEYWORDS | METH_VARARGS, Host_onCreateBuffers_doc},
	{"onDestroyBuffers", (PyCFunction)Host_onDestroyBuffers, METH_NOARGS, Host_onDestroyBuffers_doc},
	{"onPlay", (PyCFunction)Host_onPlay, METH_NOARGS, Host_onPlay_doc},
	{"onStop", (PyCFunction)Host_onStop, METH_NOARGS, Host_onStop_doc},
	{NULL}  /* Sentinel */
};

PyTypeObject Host_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /*	tp_name :			*/	"unit.cuda.Host",
    /*	tp_basicsize :		*/	sizeof(Host),
    /*	tp_itemsize :		*/	0,
    /*	tp_dealloc :		*/	(destructor)Host_dealloc,
    /*	tp_print :			*/	0,
    /*	tp_getattr :		*/	0,
    /*	tp_setattr :		*/	0,
    /*	tp_reserved :		*/	0,
    /*	tp_repr :			*/	0,
    /*	tp_as_number :		*/	0,
    /*	tp_as_sequence :	*/	0,
    /*	tp_as_mapping :		*/	0,
    /*	tp_hash :			*/	0,
    /*	tp_call :			*/	0,
    /*	tp_str :			*/	0,
    /*	tp_getattro :		*/	0,
    /*	tp_setattro :		*/	0,
    /*	tp_as_buffer :		*/	0,
    /*	tp_flags :			*/	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*	tp_doc :			*/	Host_doc,
    /*	tp_traverse :		*/	0,
    /*	tp_clear :			*/	0,
    /*	tp_richcompare :	*/	0,
    /*	tp_weaklistoffset :	*/	0,
    /*	tp_iter :			*/	0,
    /*	tp_iternext :		*/	0,
    /*	tp_methods :		*/	Host_methods,             
    /*	tp_members :		*/	Host_members,
    /*	tp_getset :			*/	Host_getseters,
    /*	tp_base :			*/	0,
    /*	tp_dict :			*/	0,
    /*	tp_descr_get		*/	0,
    /*	tp_descr_set:		*/	0,
    /*	tp_dictoffset :		*/	0,
    /*	tp_init :			*/	(initproc)Host_init,
    /*	tp_alloc :			*/	0,
	/*  tp_new :			*/	0
};

/***************************************************************************************************
*
***************************************************************************************************/

static PyMemberDef Unit_members[] = {
	{"vendor", T_STRING_INPLACE, offsetof(Unit, setup.info.vendor), READONLY, "vendor"},
	{"name", T_STRING_INPLACE, offsetof(Unit, setup.info.name), READONLY, "name"},
	{"version", T_STRING_INPLACE, offsetof(Unit, setup.info.version), READONLY, "version"},
	{"description", T_STRING_INPLACE, offsetof(Unit, setup.info.description), READONLY, "description"},
	{NULL}  /* Sentinel */
};

static PyGetSetDef Unit_getseters[] = {
//    {"first", (getter)Noddy_getfirst, (setter)Noddy_setfirst, "first name", NULL},
//    {"last", (getter)Noddy_getlast, (setter)Noddy_setlast, "last name", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Unit_methods[] = {
	/* Sentinel */	{NULL}  
};

PyTypeObject Unit_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /*	tp_name :			*/	"unit.cuda.Unit",
    /*	tp_basicsize :		*/	sizeof(Unit),
    /*	tp_itemsize :		*/	0,
    /*	tp_dealloc :		*/	(destructor)Unit_dealloc,
    /*	tp_print :			*/	0,
    /*	tp_getattr :		*/	0,
    /*	tp_setattr :		*/	0,
    /*	tp_reserved :		*/	0,
    /*	tp_repr :			*/	0,
    /*	tp_as_number :		*/	0,
    /*	tp_as_sequence :	*/	0,
    /*	tp_as_mapping :		*/	0,
    /*	tp_hash :			*/	0,
    /*	tp_call :			*/	0,
    /*	tp_str :			*/	0,
    /*	tp_getattro :		*/	0,
    /*	tp_setattro :		*/	0,
    /*	tp_as_buffer :		*/	0,
    /*	tp_flags :			*/	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*	tp_doc :			*/	Unit_doc,
    /*	tp_traverse :		*/	0,
    /*	tp_clear :			*/	0,
    /*	tp_richcompare :	*/	0,
    /*	tp_weaklistoffset :	*/	0,
    /*	tp_iter :			*/	0,
    /*	tp_iternext :		*/	0,
    /*	tp_methods :		*/	Unit_methods,             
    /*	tp_members :		*/	Unit_members,
    /*	tp_getset :			*/	Unit_getseters,
    /*	tp_base :			*/	0,
    /*	tp_dict :			*/	0,
    /*	tp_descr_get		*/	0,
    /*	tp_descr_set:		*/	0,
    /*	tp_dictoffset :		*/	0,
    /*	tp_init :			*/	(initproc)Unit_init,
    /*	tp_alloc :			*/	0,
	/*  tp_new :			*/	0
};

/***************************************************************************************************
*
***************************************************************************************************/
#if 0
static PyMemberDef UnitKernel_members[] = {
	{NULL}  /* Sentinel */
};

static PyGetSetDef UnitKernel_getseters[] = {
    {NULL}  /* Sentinel */
};

static PyMethodDef UnitKernel_methods[] = {
	{NULL}  /* Sentinel */
};

PyTypeObject UnitKernel_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /*	tp_name :			*/	"unit.cuda.UnitKernel",
    /*	tp_basicsize :		*/	sizeof(UnitKernel),
    /*	tp_itemsize :		*/	0,
    /*	tp_dealloc :		*/	(destructor)UnitKernel_dealloc,
    /*	tp_print :			*/	0,
    /*	tp_getattr :		*/	0,
    /*	tp_setattr :		*/	0,
    /*	tp_reserved :		*/	0,
    /*	tp_repr :			*/	0,
    /*	tp_as_number :		*/	0,
    /*	tp_as_sequence :	*/	0,
    /*	tp_as_mapping :		*/	0,
    /*	tp_hash :			*/	0,
    /*	tp_call :			*/	0,
    /*	tp_str :			*/	0,
    /*	tp_getattro :		*/	0,
    /*	tp_setattro :		*/	0,
    /*	tp_as_buffer :		*/	0,
    /*	tp_flags :			*/	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*	tp_doc :			*/	UnitKernel_doc,
    /*	tp_traverse :		*/	0,
    /*	tp_clear :			*/	0,
    /*	tp_richcompare :	*/	0,
    /*	tp_weaklistoffset :	*/	0,
    /*	tp_iter :			*/	0,
    /*	tp_iternext :		*/	0,
    /*	tp_methods :		*/	UnitKernel_methods,             
    /*	tp_members :		*/	UnitKernel_members,
    /*	tp_getset :			*/	UnitKernel_getseters,
    /*	tp_base :			*/	0,
    /*	tp_dict :			*/	0,
    /*	tp_descr_get		*/	0,
    /*	tp_descr_set:		*/	0,
    /*	tp_dictoffset :		*/	0,
    /*	tp_init :			*/	(initproc)UnitKernel_init,
    /*	tp_alloc :			*/	0,
	/*  tp_new :			*/	0
};
#endif
/***************************************************************************************************
*
***************************************************************************************************/

static PyModuleDef cuda_module = {
    PyModuleDef_HEAD_INIT,
    "unit._cuda",
    "",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cuda(void) 
{
	CU_BEGIN( PyInit__cuda )

	CU_CALL( cuInit(0) );
	CU_CALL( cuDriverGetVersion((int*)&cuda_version) );
	CU_CALL( cuDeviceGetCount((int*)&cuda_device_count) );

	CU_ASSERT( cuda_device_count, NULL, "This machine not have computation devices");

	cuda_device = new CudaDevice[cuda_device_count];

	for(unsigned long devidx = 0; devidx < cuda_device_count; devidx++)
	{
		CudaDevice* device = cuda_device + devidx;
		CU_CALL( cuDeviceGet(&device->handle, devidx) );
		CU_CALL( cuDeviceGetName(device->name, sizeof(device->name), devidx) );
		CU_CALL( cuDeviceGetProperties(&device->prop, devidx) );
	}

    PyObject* m = PyModule_Create(&cuda_module);
    if (m == NULL)
        return NULL;
	
	//	CudaException
	cuda_exception = PyErr_NewException("unit._cuda.CudaException", NULL, NULL);
	Py_INCREF(cuda_exception);
	PyModule_AddObject(m, "CudaException", cuda_exception);

	//	Host
	Host_type.tp_new = PyType_GenericNew;
	if( PyType_Ready(&Host_type) < 0 )
        return NULL;

    Py_INCREF(&Host_type);
	PyModule_AddObject(m, "Host", (PyObject *)&Host_type);

	//	Unit	
	Unit_type.tp_new = PyType_GenericNew;
	if (PyType_Ready(&Unit_type) < 0)
        return NULL;

    Py_INCREF(&Unit_type);
	PyModule_AddObject(m, "Unit", (PyObject *)&Unit_type);
	/*
	//	Unit	
	UnitKernel_type.tp_new = PyType_GenericNew;
	if (PyType_Ready(&UnitKernel_type) < 0)
        return NULL;

    Py_INCREF(&UnitKernel_type);
	PyModule_AddObject(m, "UnitKernel", (PyObject *)&UnitKernel_type);
	*/

	return m;

	CU_END( NULL )
}
