#include "DSP.h"
#include <stdio.h>
#include <math.h>


UNIT("Prodisign", "test", "0.1beta", "");

GLOBAL HostBuffer* out;
GLOBAL HostBuffer* in;
GLOBAL HostBuffer* wave;

GLOBAL float freq;

EXPORT void onLoad()
{
	DEBUG("Waveform.onLoad\n");
}

EXPORT void onUnload()
{
	DEBUG("WAveform.onUnload\n");
}

EXPORT void onCreateBuffers()
{
	DEBUG("Waveform.onCreateBuffers sample_rate: %d\n", setup.host->sample_rate);

	out = setup.host->io.getOut(0);
	in = setup.host->io.getIn(0);

	//	aloja un buffer con el samplerate de longitud
	wave = setup.host->io.allocBuffer();
	wave->length = setup.host->sample_rate;
	wave->format = HOST_BUFFER_FORMAT_INT32;
	
	freq = 1.0 / setup.host->chunk_length;

	setup.process.launch("onProcess").block(setup.host->chunk_length);
}

EXPORT void onDestroyBuffers()
{
	DEBUG("waveform.onDestroyBuffers\n");
}

EXPORT void onPlay()
{
	DEBUG("Waveform.onPlay\n");
	DEBUG("  wave buffer ptr: %p\n", wave->ptr);
	DEBUG("  out buffer ptr: %p\n", out->ptr);
	DEBUG("  in buffer ptr: %p\n", in->ptr);
}

#define PI 3.1415


EXPORT void onStop()
{
	DEBUG("Waveform.onStop\n");
}

EXPORT void onProcess()
{
	unsigned long s = threadIdx.x;
	out[0].i32[s] = int( sin(2 * PI * s * freq) * 0x7FFFFFFF );
	wave[0].i32[s] = in[0].i32[s];
}
