#pragma once

#include <Windows.h>

#undef _DEBUG
#include <Python.h>
#include <structmember.h>

#include "asiosys.h"
#include "asio.h"
#include "iasiodrv.h"

typedef void (OnProcess)(PyObject*, void**, void**, void(*)());

typedef struct
{
    PyObject_HEAD

	IASIO* iasio;

	char name[32];

	long version;

	// ASIO->GetSampleRate
	double sample_rate;

	// ASIO->getBufferSize
	long min_buffer_length;
	long max_buffer_length;
	long preferred_buffer_length;
	long buffer_granularity;

	//	ASIO->getChannels
	long input_channel_count;
	long output_channel_count;

	ASIOChannelInfo* input_channel_info;
	ASIOChannelInfo* output_channel_info;

	// ASIO->getLatencies
	long input_latency;
	long output_latency;

	long buffer_length;
	long buffer_count;
	long input_buffer_count;
	long output_buffer_count;

	ASIOBufferInfo* buffer_info;
	ASIOBufferInfo* input_buffer_info;
	ASIOBufferInfo* output_buffer_info;

	void** input_buffer[2];
	void** output_buffer[2];
	long slot;
	
	char is_playing;

	PyObject* onProcess;

	PyObject *host;
	OnProcess* host_onProcess;

//	Error error;
//	char error_message[256];

} Driver;


// ASIO Callbacks
template <unsigned SLOT>
static void cbBufferSwitch(long index, ASIOBool process_now);

template <unsigned SLOT> 
static ASIOTime *cbBufferSwitchTime(ASIOTime *timeInfo, long index, ASIOBool process_now);

template <unsigned SLOT>
static void cbSampleRate(ASIOSampleRate sample_rate);

template <unsigned SLOT>
static long cbMessage(long selector, long value, void* message, double* opt);