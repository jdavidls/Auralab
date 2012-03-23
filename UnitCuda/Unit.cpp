#include "Unit.h"
#include "Host.h"

#define CU_SETUP_ZERO(VAR)											\
	memset(&self->setup.VAR, 0, sizeof(self->setup.VAR));

#define CU_SETUP_DTOH(VAR)											\
	CU_CALL( cuMemcpyDtoH(											\
		&self->setup.VAR,											\
		self->setup_ptr + offsetof(UnitSetup, VAR),					\
		sizeof(self->setup.VAR)) );		

#define CU_SETUP_HTOD(VAR)											\
	CU_CALL( cuMemcpyHtoD(											\
		self->setup_ptr + offsetof(UnitSetup, VAR),					\
		&self->setup.VAR,											\
		sizeof(self->setup.VAR)) );		

#define CU_LAUNCH(KERNEL)											\
CU_CALL( cuLaunchKernel(self->KERNEL, 1, 1, 1, 1, 1, 1, 1024, 0, 0, 0) );

char* Unit_doc = "AuraLab.cuda.Unit\n"
	"";

/***************************************************************************************************
*
***************************************************************************************************/

int Unit_init(Unit* self, PyObject* args, PyObject* kwds)
{
	CU_BEGIN( Unit_init )

    static char *kwlist[] = {"host", "filename", NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwds, "Os", kwlist,
			&self->host,
			&self->filename) )
        return -1; 

	// set CUDA context from host?
	CU_CALL( cuCtxPushCurrent(self->host->cuda_context) )

	// Load Cuda Module
	CU_CALL( cuModuleLoad(&self->module, self->filename) );

	// Reads UnitSetup
	CU_CALL( cuModuleGetGlobal(&self->setup_ptr, &self->setup_bytes, self->module, "setup") );
	CU_ASSERT( self->setup_bytes == sizeof(UnitSetup), -1, "UnitSetup size don't match");
	CU_CALL( cuMemcpyDtoH(&self->setup, self->setup_ptr, sizeof(UnitSetup)) );

	// Set host setup pointer into unit setup
	self->setup.host = self->host->setup_ptr;
	self->setup.module = self->module;

	CU_SETUP_HTOD( host );
	CU_SETUP_HTOD( module );


	//	Gets Unit functions
	CU_CALL( cuModuleGetFunction(&self->onLoad, self->module, "onLoad") );
	CU_CALL( cuModuleGetFunction(&self->onUnload, self->module, "onUnload") );
	CU_CALL( cuModuleGetFunction(&self->onCreateBuffers, self->module, "onCreateBuffers") );
	CU_CALL( cuModuleGetFunction(&self->onDestroyBuffers, self->module, "onDestroyBuffers") );
	CU_CALL( cuModuleGetFunction(&self->onPlay, self->module, "onPlay") );
	CU_CALL( cuModuleGetFunction(&self->onStop, self->module, "onStop") );

	// Call unit.onLoad
	CU_LAUNCH( onLoad );
	
	//	finish
	CU_CALL( cuCtxPopCurrent(0) );

	return 0;

	CU_END( -1, cuCtxPopCurrent(0) )
}

/***************************************************************************************************
*
***************************************************************************************************/

void Unit_dealloc(Unit* self)
{
	CU_BEGIN( Unit_dealloc );

	//
	DEQUE_UNLINK(self->host, chain, self);
	
	if( self->module )
	{
		CU_CALL( cuCtxPushCurrent(self->host->cuda_context) );

		CU_LAUNCH( onUnload );

		CU_CALL( cuModuleUnload(self->module) );

		CU_CALL( cuCtxPopCurrent(0) );
	}

	return;

	CU_END( (void)0, cuCtxPopCurrent(0) );
}

/***************************************************************************************************
*
***************************************************************************************************/

PyObject* Unit_unlink(Unit* self)
{
	if( !DEQUE_NOT_LINKED_IN(self->host, chain, self) )
	{
		Py_DECREF(self);
		DEQUE_UNLINK(self->host, chain, self);
	}

	return Py_None;
}


/***************************************************************************************************
*
***************************************************************************************************/

#define DBG_INT(X)	printf(#X " = %d\n", X)
void Unit_launchQueue(Unit* self)
{
	CU_BEGIN( Unit_launchQueue );
	
	CU_SETUP_DTOH( queue );

	for(unsigned long i = 0; i < self->setup.queue.length; i++)
	{
		UnitKernel *k = self->setup.queue.kernel + i;
		CU_CALL( cuModuleGetFunction(&k->function_ptr, self->module, k->function_name) );

		if( k->shared_size > self->host->device->prop.sharedMemPerBlock )
			k->shared_size = self->host->device->prop.sharedMemPerBlock;

		for(unsigned long i = 0; i < k->arg_length; k->args[++i] = 0)
			k->args[i] = k->arg_data + (unsigned long) k->args[i];

		CU_CALL( cuLaunchKernel(k->function_ptr,
			k->gridDim.x, k->gridDim.y, k->gridDim.z,
			k->blockDim.x, k->blockDim.y, k->blockDim.z,
			k->shared_size, 0, k->args, 0) );
	}

	CU_SETUP_ZERO( queue );
	CU_SETUP_HTOD( queue );
	
	return;

	CU_END( (void)0 );
}


/***************************************************************************************************
*
***************************************************************************************************/

void Unit_prepareProcess(Unit* self)
{
	CU_BEGIN( Unit_prepareProcess );
	
	CU_SETUP_DTOH( process );
	
	for(unsigned long i = 0; i < self->setup.process.length; i++)
	{
		UnitKernel *k = self->setup.process.kernel + i;
		CU_CALL( cuModuleGetFunction(&k->function_ptr, self->module, k->function_name) );

		if( k->shared_size > self->host->device->prop.sharedMemPerBlock )
			k->shared_size = self->host->device->prop.sharedMemPerBlock;

		for(unsigned long i = 0; i < k->arg_length; k->args[++i] = 0)
			k->args[i] = k->arg_data + (unsigned long) k->args[i];

		//printf("  %s\n",  k->function_name);
	}
	
	return;
	
	CU_END( (void)0 );
}


/***************************************************************************************************
*
***************************************************************************************************/

void Unit_launchProcess(Unit* self)
{
	CU_BEGIN( Unit_launchProcess );
	
	for(unsigned long i = 0; i < self->setup.process.length; i++)
	{
		UnitKernel *k = self->setup.process.kernel + i;

		//printf("launch %s\n", k->function_name);

		CU_CALL( cuLaunchKernel(k->function_ptr,
			k->gridDim.x, k->gridDim.y, k->gridDim.z,
			k->blockDim.x, k->blockDim.y, k->blockDim.z,
			k->shared_size, 0, k->args, 0) );
	}
	
	return;

	CU_END( (void)0 );
}
