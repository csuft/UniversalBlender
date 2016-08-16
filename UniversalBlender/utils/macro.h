
#ifndef CODEC_LIB_MACRO_H
#define CODEC_LIB_MACRO_H

#define CDL_VERSION "1.1.0"

//#define AAC_DEC

#define CDL_BETA

/* config */

#define CDL_CFG_CODE               "code"
#define CDL_CFG_CODE_PROFILE       "profile"
#define CDL_CFG_CODE_PRESET        "preset"
#define CDL_CFG_CODE_TUNE          "tune"
#define CDL_CFG_CODE_RCMODE        "rcmode"
#define CDL_CFG_CODE_QP            "qp"
#define CDL_CFG_CODE_RF            "rf"
#define CDL_CFG_CODE_RATIO         "ratio"
#define CDL_CFG_CODE_BITRATE       "bitrate"

#define CDL_CFG_HLS               "hls"
#define CDL_CFG_HLS_DURATION      "duration"
#define CDL_CFG_HLS_LISTSIZE      "listsize"
#define CDL_CFG_HLS_LISTNAME      "listname"

#define CDL_CFG_LOG               "log"
#define CDL_CFG_LOG_ENABLE        "enable"
#define CDL_CFG_LOG_LEVEL         "level"
#define CDL_CFG_LOG_SIZE          "size"
#define CDL_CFG_LOG_STDOUT        "stdout"

#define CDL_CFG_DBG               "debug"
#define CDL_CFG_DBG_SRCH264       "srch264"
#define CDL_CFG_DBG_DSTH264       "dsth264"
#define CDL_CFG_DBG_SRCYUV        "srcyuv"
#define CDL_CFG_DBG_DSTYUV        "dstyuv"

#define CDL_SUCCESS 200

#define CDL_MEDIA_AUDIO 0
#define CDL_MEDIA_VIDEO 1

#define CDL_MP4_VIDEO_BUFF_LEN 1024*1024*3

/* function */
#define CDL_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define CDL_MIN(a,b) (((a) < (b)) ? (a) : (b))

#define CDL_DELETE_MIN(p) \
if (p) \
{\
	delete p;\
}

#define CDL_DELETE(p) \
if (p) \
{\
	delete p;\
	p = NULL;\
}

#define CDL_DELETE_ARRAY(p) \
if (p) \
{\
	delete[] p;\
	p = NULL;\
}

#define CDL_FREE(p) \
if (p) \
{\
	free(p);\
	p = NULL;\
}

#if (defined _WIN32 || defined _WIN64)
#define CDL_STRCMP(src, dst) _stricmp(src, dst)
#define CDL_SPRINTF _snprintf
#else
#define CDL_STRCMP(src, dst) strcmp(src, dst)
#define CDL_SPRINTF snprintf
#endif


#endif

