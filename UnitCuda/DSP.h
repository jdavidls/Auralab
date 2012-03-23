#pragma once
#include <cuda.h>
#include <vector_types.h>

#ifdef BUILDING_HOST
#define EXPORT
#define GLOBAL
#define LOCAL
//#define CONST
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define EXPORT	extern "C" __global__
#define GLOBAL	__device__
#define LOCAL	__shared__
#define CONST	__constant__
#endif

#define GLOBALS_MAX_COUNT				(128)
#define GLOBALS_NAME_MAX_LENGTH			(24)
#define KERNEL_NAME_MAX_LENGTH			(32)

//----------------------------------------------------------------------------------- Buffer Format

#define HOST_BUFFER_FORMAT(TYPE, BITS)		( (TYPE) | ((BITS) << 16) )
#define HOST_BUFFER_FORMAT_BITS(FORMAT)		( (FORMAT) >> 16 )
#define HOST_BUFFER_FORMAT_BYTES(FORMAT)	( (FORMAT) >> 19 )
#define HOST_BUFFER_FORMAT_TYPE(FORMAT)		( (FORMAT) & 0xffff )

#define HOST_BUFFER_FORMAT_INT32			HOST_BUFFER_FORMAT('i', 32)
#define HOST_BUFFER_FORMAT_INT64			HOST_BUFFER_FORMAT('i', 64)
#define HOST_BUFFER_FORMAT_FLOAT32			HOST_BUFFER_FORMAT('f', 32)
#define HOST_BUFFER_FORMAT_FLOAT64			HOST_BUFFER_FORMAT('f', 64)

//-------------------------------------------------------------------------------------- Unit Decls

#ifndef BUILDING_HOST	/* FOR CUDA UNITS*/

#define DEBUG printf

#define DBG_VAL(VAL, FMT)	printf("\t" #VAL ": \t\t" FMT "\n", VAL)


#define UNIT(VENDOR, NAME, VERSION, DESCRIPTION)												\
GLOBAL UnitSetup setup =																		\
{																								\
	{VENDOR, NAME, VERSION, DESCRIPTION}														\
};

GLOBAL void str_copy(char* dst, const char* src)
{
	while( *(dst++) = *(src++) );
}

#endif


/***************************************************************************************************
*
***************************************************************************************************/

struct HostBuffer
{
	unsigned long length;					//	(Host<-Device):minimal length (Device->Host): optimal length
	unsigned long bytes;					//	(Host->Device)
	unsigned long offset;					//	(Host->Device)
	unsigned long mask;						//	(Host->Device)

	unsigned long format;					//	(Host->Device)

#ifdef BUILDING_HOST
	CUdeviceptr ptr;						//	(Host->Device)
#else
	union
	{
		void *ptr;
		long *i32;
		long long *i64;
		float *f32;
		double *f64;
	};
#endif

};

struct HostIO
{
	unsigned long count;					//	(Host->Device)
	unsigned long outputs;					//
	unsigned long inputs;					//
	HostBuffer buffer[64];					//	(Host<->Device)
};


/***************************************************************************************************
*
***************************************************************************************************/

struct HostSetup
{
	CUdevprop_st device_info;				//	(Host->Device)
	unsigned long sample_rate;				//  (Host->Device)
	unsigned long chunk_length;				//	(Host->Device)

	HostIO io;
};

/***************************************************************************************************
*
***************************************************************************************************/

struct UnitVar
{
	char name[GLOBALS_NAME_MAX_LENGTH];

#ifdef BUILDING_HOST
	CUdeviceptr ptr;						//	(Host->Device)
#else
	void* ptr;
#endif

	/*
	...fills python buffer data here...
	*/
	

	unsigned long format;

#ifndef BUILDING_HOST

	template<class T>
	void set(char* name, T& var);
	

	template<>
	GLOBAL void set<>(char* name, int& var)
	{
		str_copy(this->name, name);
		ptr = &var;
	}

#endif

};

struct UnitVars
{
	unsigned long length;
	UnitVar vars[GLOBALS_MAX_COUNT];

#ifndef BUILDING_HOST

	template<class T>
	GLOBAL void reg(char* name, T& var)
	{
		vars[length++].set(name, var);
	}

#endif

};

/***************************************************************************************************
*
***************************************************************************************************/



struct UnitKernel
{
	char function_name[32];	//	(Host<-Device)
	uint3 blockDim;								//	(Host<-Device)
	uint3 gridDim;								//	(Host<-Device)
	unsigned long shared_size;						//	(Host<-Device)

#ifdef BUILDING_HOST
	CUfunction function_ptr;					//	(Host->Device)
	void* args[16];
#else
	void* function_ptr;
	unsigned long arg_offset[16];	
#endif
	unsigned long arg_length;
	unsigned char arg_data[512];

#ifndef BUILDING_HOST

	GLOBAL UnitKernel &block(unsigned int x, unsigned int y = 1, unsigned int z = 1)
	{
		this->blockDim.x = x;
		this->blockDim.y = y;
		this->blockDim.z = z;
		return *this;
	}

	GLOBAL UnitKernel &grid(unsigned int x, unsigned int y = 1, unsigned int z = 1)
	{
		this->gridDim.x = x;
		this->gridDim.y = y;
		this->gridDim.z = z;
		return *this;
	}

	GLOBAL UnitKernel &shared(unsigned long shared_size)
	{
		this->shared_size = shared_size;
		return *this;
	}

	template<class T>
	GLOBAL UnitKernel &arg(T val)
	{
		unsigned long offset = arg_offset[arg_length++];
		*(T*)(arg_data + offset) = val;
		arg_offset[arg_length] = offset + sizeof(T);
		return *this;
	}

#endif

};

struct UnitLauncher
{
	unsigned long length;					//	(Host<-Device)
	UnitKernel kernel[64];					//	(Host<-Device)

#ifndef BUILDING_HOST

	GLOBAL UnitKernel &launch(char* function_name)
	{
		UnitKernel* k = kernel + length++;
		str_copy(k->function_name, function_name);
		k->blockDim = dim3();
		k->gridDim = dim3();
		k->shared_size = 0xFFFFFFFF;
		k->arg_length = 0;
		k->arg_offset[0] = 0;
		return *k;
	}

#endif
};

/***************************************************************************************************
*
***************************************************************************************************/

struct UnitInfo
{
	char vendor[32];						//	(Host<-Unit)
	char name[32];							//	(Host<-Unit)
	char version[8];						//	(Host<-Unit)
	char description[56];					//	(Host<-Unit)
};

struct UnitSetup
{
	UnitInfo info;							//	(Host<-Unit)
	UnitVars globals;						//	(Host<-Unit)
	
	UnitLauncher queue;					//	(Host<-Device)
	UnitLauncher process;					//	(Host<-Device)

	CUmodule module;						//	(Host->Unit)

#ifdef BUILDING_HOST
	CUdeviceptr host;						//	(Host->Unit)
#else
	HostSetup* host;
#endif

};
