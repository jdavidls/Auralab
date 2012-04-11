#include "DSP.h"
#include <stdio.h>
#include <math.h>


UNIT("Prodisign", "test", "0.1beta", "");

GLOBAL HostBuffer* out;
GLOBAL HostBuffer* in;

GLOBAL float freq;

EXPORT void onLoad()
{
	DEBUG("onLoad\n");
}

EXPORT void onUnload()
{
	DEBUG("onUnload\n");
}

EXPORT void onCreateBuffers()
{
	DEBUG("onCreateBuffers sample_rate: %d\n", setup.host->sample_rate);

	freq = 1.0 / setup.host->chunk_length;
	
	setup.process.launch("onProcess").block(setup.host->chunk_length);

}

EXPORT void onDestroyBuffers()
{
	DEBUG("onDestroyBuffers\n");
}

EXPORT void onPlay()
{
	DEBUG("onPlay\n");
	
	out = setup.host->io.buffer;
	DEBUG("  buffer ptr: %p\n", out->ptr);
}

#define PI 3.1415


EXPORT void onStop()
{
	DEBUG("onStop\n");
}


EXPORT void onProcess()
{
	unsigned long s = threadIdx.x;
	out[0].i32[s] = int( sin(2 * PI * s * freq) * 0x7FFFFFFF );
}
