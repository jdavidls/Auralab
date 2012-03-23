#include <Python.h>
#include <structmember.h>

#include "Waveform.h"



//--------------------------------------------------------------------------------------------------
static PyMemberDef Waveform_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef Waveform_getseters[] = {
//    {"first", (getter)Noddy_getfirst, (setter)Noddy_setfirst, "first name", NULL},
//    {"last", (getter)Noddy_getlast, (setter)Noddy_setlast, "last name", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Waveform_methods[] = {
	{NULL}  /* Sentinel */
};

static PyTypeObject asio_driver_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "unit.waveform.Waveform",   /* tp_name */
    sizeof(Waveform),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)Waveform_dealloc, /* tp_dealloc */
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
    "Waveform",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Waveform_methods,             /* tp_methods */
    Waveform_members,             /* tp_members */
    Waveform_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Waveform_init,      /* tp_init */
    0,                         /* tp_alloc */
    0//	Noddy_new,                 /* tp_new */
};

static PyModuleDef asio_module = {
    PyModuleDef_HEAD_INIT,
    "unit._waveform",
    "Example module that creates an extension type.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

//--------------------------------------------------------------------------------------------------
PyMODINIT_FUNC PyInit__asio(void) 
{
    PyObject* m;

    asio_driver_type.tp_new = PyType_GenericNew;

	if (PyType_Ready(&asio_driver_type) < 0)
        return NULL;

    m = PyModule_Create(&asio_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&asio_driver_type);
	PyModule_AddObject(m, "Driver", (PyObject *)&asio_driver_type);

	return m;
}
