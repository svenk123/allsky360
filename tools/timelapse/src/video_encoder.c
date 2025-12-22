/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include "video_encoder.h"
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#if DEBUG
#include <libavutil/log.h>
#endif

static AVFormatContext *fmt_ctx;
static AVCodecContext *codec_ctx;
static AVStream *video_stream;
static AVFrame *frame = NULL;
static struct SwsContext *sws_ctx;
static int frame_counter = 0;
static int width, height;

int init_encoder_nvenc(const char *filename, int w, int h, int fps) {
  av_log_set_level(AV_LOG_WARNING);
#if DEBUG
  av_log_set_level(AV_LOG_DEBUG);
#endif

#if LIBAVFORMAT_VERSION_MAJOR < 58
  av_register_all();
#endif

  const char *format = strstr(filename, ".mkv")   ? "matroska"
                       : strstr(filename, ".avi") ? "avi"
                       : strstr(filename, ".mov") ? "mov"
                       : strstr(filename, ".mp4") ? "mp4"
                                                  : NULL;

  if (avformat_alloc_output_context2(&fmt_ctx, NULL, format, filename) < 0 ||
      !fmt_ctx) {
    fprintf(stderr, "[ERROR] Failed to create output context for file: %s\n",
            filename);
    return -1;
  }

  AVCodec *codec = avcodec_find_encoder_by_name("libx264");
  if (!codec) {
    fprintf(stderr, "[ERROR] libx264 encoder not found!\n");
    return -1;
  }

  video_stream = avformat_new_stream(fmt_ctx, NULL);
  if (!video_stream) {
    fprintf(stderr, "[ERROR] Failed to create video stream.\n");
    return -1;
  }

  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    fprintf(stderr, "[ERROR] Failed to allocate codec context.\n");
    return -1;
  }

  // Video size
  width = w;
  height = h;
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->bit_rate = 4000000;

  // Set framerate and timebase
  AVRational fps1 = {fps, 1};
  AVRational time_base = {1, fps}; // 1 Tick = 1 Frame

  codec_ctx->framerate = fps1;
  codec_ctx->time_base = time_base;

  // Encoder tuning
  codec_ctx->gop_size = 10;
  codec_ctx->max_b_frames = 1;
  codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

  if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
    codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

  video_stream->time_base = codec_ctx->time_base;
  video_stream->avg_frame_rate = (AVRational){fps, 1};

  // Libx264 options
  AVDictionary *opts = NULL;
  av_dict_set(&opts, "preset", "slow", 0);
  av_dict_set(&opts, "tune", "zerolatency", 0);

  if (avcodec_open2(codec_ctx, codec, &opts) < 0) {
    fprintf(stderr, "[ERROR] Failed to open encoder.\n");
    av_dict_free(&opts);
    return -1;
  }
  av_dict_free(&opts);

  if (avcodec_parameters_from_context(video_stream->codecpar, codec_ctx) < 0) {
    fprintf(stderr, "[ERROR] Failed to copy codec parameters.\n");
    return -1;
  }

  video_stream->time_base = codec_ctx->time_base;
  video_stream->avg_frame_rate = codec_ctx->framerate;

  if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&fmt_ctx->pb, filename, AVIO_FLAG_WRITE) < 0) {
      fprintf(stderr, "[ERROR] Failed to open output file: %s\n", filename);
      return -1;
    }
  }

  if (avformat_write_header(fmt_ctx, NULL) < 0) {
    fprintf(stderr, "[ERROR] Failed to write stream header.\n");
    return -1;
  }

  av_dump_format(fmt_ctx, 0, filename, 1);

  // Color conversion
  // SWS_BILINEAR = Quality: average, Speed: average
  // SWS_FAST_BILINEAR = Quality: low to average, Speed: high
  sws_ctx =
      sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height,
                     AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL);
  if (!sws_ctx) {
    fprintf(stderr, "[ERROR] Failed to initialize swscale context.\n");
    return -1;
  }

  frame = av_frame_alloc();
  if (!frame) {
    fprintf(stderr, "Failed to allocate reusable AVFrame\n");
    return -1;
  }

  frame->format = codec_ctx->pix_fmt;
  frame->width = codec_ctx->width;
  frame->height = codec_ctx->height;

  if (av_frame_get_buffer(frame, 32) < 0) {
    fprintf(stderr, "Failed to allocate reusable frame buffer\n");
    return -1;
  }

  return 0;
}

void encode_frame(uint8_t *rgb_data) {
#if DEBUG
  printf("[INFO] CodecContext: time_base=%d/%d, framerate=%d/%d\n",
         codec_ctx->time_base.num, codec_ctx->time_base.den,
         codec_ctx->framerate.num, codec_ctx->framerate.den);

  printf("[INFO] Stream time_base: %d/%d avg_frame_rate: %d/%d\n",
         video_stream->time_base.num, video_stream->time_base.den,
         video_stream->avg_frame_rate.num, video_stream->avg_frame_rate.den);
#endif

  frame->pts = frame_counter;

  int ret = av_frame_make_writable(frame);
  if (ret < 0) {
    fprintf(stderr, "Frame not writable\n");
    return;
  }

  uint8_t *in_data[1] = {rgb_data};
  int in_linesize[1] = {3 * width};

  if (!sws_ctx) {
    fprintf(stderr, "[ERROR] sws_ctx not initialized.\n");
    return;
  }

  sws_scale(sws_ctx, (const uint8_t *const *)in_data, in_linesize, 0, height,
            frame->data, frame->linesize);

  if (avcodec_send_frame(codec_ctx, frame) < 0) {
    fprintf(stderr, "[ERROR] Failed to send frame to encoder.\n");
    return;
  }

  AVPacket *pkt = av_packet_alloc();
  if (!pkt) {
    fprintf(stderr, "[ERROR] Failed to allocate AVPacket.\n");
    return;
  }

  while (avcodec_receive_packet(codec_ctx, pkt) == 0) {
    pkt->stream_index = video_stream->index;
    av_packet_rescale_ts(pkt, codec_ctx->time_base, video_stream->time_base);
    av_write_frame(fmt_ctx, pkt);
    av_packet_unref(pkt);

    frame_counter++;
  }

  av_packet_free(&pkt);
}

void finalize_encoder(void) {
  // Flushing: send NULL to signal last frame
  if (avcodec_send_frame(codec_ctx, NULL) < 0) {
    fprintf(stderr, "[ERROR] Failed to flush encoder.\n");
    return;
  }

  AVPacket *pkt = av_packet_alloc();
  if (!pkt) {
    fprintf(stderr,
            "[ERROR] Failed to allocate AVPacket in finalize_encoder.\n");
    return;
  }

  // Write stream as long as the encoder puts packages
  while (avcodec_receive_packet(codec_ctx, pkt) == 0) {
    pkt->stream_index = video_stream->index;
    av_packet_rescale_ts(pkt, codec_ctx->time_base, video_stream->time_base);
    av_write_frame(fmt_ctx, pkt);
    frame_counter++;
    av_packet_unref(pkt);
  }

  av_packet_free(&pkt);

  // Write trailer
  av_write_trailer(fmt_ctx);

  // Free resources
  avcodec_free_context(&codec_ctx);
  if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE))
    avio_closep(&fmt_ctx->pb);
  avformat_free_context(fmt_ctx);

  av_frame_free(&frame);

  printf("[FINAL] Encoded %d frames\n", frame_counter);
}
