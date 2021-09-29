![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

**Discord** invite link for for communication and questions: https://discord.gg/zSq8rtW

## Scaled-YOLOv4: 

* **paper:** https://arxiv.org/abs/2011.08036

* **source code - Pytorch (use to reproduce results):** https://github.com/WongKinYiu/ScaledYOLOv4

* **source code - Darknet:** https://github.com/AlexeyAB/darknet

* **Medium:** https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982?source=friends_link&sk=c8553bfed861b1a7932f739d26f487c8

## YOLOv4:

* **paper:** https://arxiv.org/abs/2004.10934

* **source code:** https://github.com/AlexeyAB/darknet

* **Wiki:** https://github.com/AlexeyAB/darknet/wiki

* **useful links:** https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

![scaled_yolov4](https://user-images.githubusercontent.com/4096485/112776361-281d8380-9048-11eb-8083-8728b12dcd55.png) AP50:95 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2011.08036

----

![YOLOv4Tiny](https://user-images.githubusercontent.com/4096485/101363015-e5c21200-38b1-11eb-986f-b3e516e05977.png)

----

![YOLOv4](https://user-images.githubusercontent.com/4096485/90338826-06114c80-dff5-11ea-9ba2-8eb63a7409b3.png)


----

![OpenCV_TRT](https://user-images.githubusercontent.com/4096485/90338805-e5e18d80-dff4-11ea-8a68-5710956256ff.png)


## Citation

```
@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection}, 
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```
#缺陷检测
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>


#include "sample_comm_nnie.h"
#include "nnie_sample_plug.h"
#include "yolov2_hand_detect.h"

#include "hi_ext_util.h"
#include "mpp_help.h"
#include "ai_plug.h"
#include "ive_img.h"
#include "hisignalling.h"

#define PLUG_UUID        
#define PLUG_DESC               // UTF8 encode

#define FRM_WIDTH          640
#define FRM_HEIGHT         384
#define MODEL_FILE_GESTURE    "./plugs/hand_gesture.wk" // darknet framework wk model

#define RET_NUM_MAX        4
#define SCORE_MAX          4096
#define THRESH_MIN         0.25
#define DETECT_OBJ_MAX     32
#define PIRIOD_NUM_MAX     49   // Logs are printed when the number of targets is detected
#define DRAW_RETC_THICK    2    // Draw the width of the line
#define IMAGE_WIDTH		   224  // The resolution of the model IMAGE sent to the classification is 224*224
#define IMAGE_HEIGHT	   224
#define OSD_FONT_WIDTH	   16
#define OSD_FONT_HEIGHT	   24
#define GESTURE_MIN        30   // confidence threshold
#define WIDTH_LIMIT        32
#define HEIGHT_LIMIT       32

static OsdSet* g_osdsGesture = NULL;
static HI_S32 g_osd0Gesture = -1;
int uart_fd;
static const char YOLO2_HAND_DETECT_RESNET_CLASSIFY[] = "{"
    "\"uuid\": " PLUG_UUID ","
    "\"desc\": " PLUG_DESC ","
    "\"frmWidth\": " HI_TO_STR(FRM_WIDTH) ","
    "\"frmHeight\": " HI_TO_STR(FRM_HEIGHT) ","
    "\"butt\": 0"
"}";

static const char* Yolo2HandDetectResnetClassifyProf(void)
{
    return YOLO2_HAND_DETECT_RESNET_CLASSIFY;
}

static HI_S32 Yolo2HandDetectResnetClassifyLoad(uintptr_t* model, OsdSet* osds)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    g_osdsGesture = osds;
    HI_ASSERT(g_osdsGesture);
    g_osd0Gesture = OsdsCreateRgn(g_osdsGesture);
    HI_ASSERT(g_osd0Gesture >= 0);

    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    HandDetectInit(); // Initialize the hand detection model
    uart_fd = uartOpenInit();
    return ret;
}

static HI_S32 Yolo2HandDetectResnetClassifyUnload(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    if (g_osdsGesture) {
        OsdsClear(g_osdsGesture);
        g_osdsGesture = NULL;
    }
    HandDetectExit(); // Uninitialize the hand detection model

    return 0;
}

/**
	Get the maximum hand
*/
static HI_S32 GetBiggestHandIndex(RectBox boxs[], int detectNum)
{
    HI_S32 handIndex = 0;
    HI_S32 biggestBoxIndex = handIndex;
    HI_S32 biggestBoxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
    HI_S32 biggestBoxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
    HI_S32 biggestBoxArea = biggestBoxWidth * biggestBoxHeight;

    for (handIndex = 1; handIndex < detectNum; handIndex++) {
        HI_S32 boxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
        HI_S32 boxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
        HI_S32 boxArea = boxWidth * boxHeight;
        if (biggestBoxArea < boxArea) {
            biggestBoxArea = boxArea;
            biggestBoxIndex = handIndex;
        }
        biggestBoxWidth = boxs[biggestBoxIndex].xmax - boxs[biggestBoxIndex].xmin + 1;
        biggestBoxHeight = boxs[biggestBoxIndex].ymax - boxs[biggestBoxIndex].ymin + 1;
    }

    if ((biggestBoxWidth == 1) || (biggestBoxHeight == 1) || (detectNum == 0)) {
        biggestBoxIndex = -1;
    }

    return biggestBoxIndex;
}

/**
    Add gesture recognition information next to the rectangle
*/
static void HandDetectAddTxt(const RectBox box, const RecogNumInfo resBuf, uint32_t color)
{
    HI_OSD_ATTR_S osdRgn;
    char osdTxt[TINY_BUF_SIZE];
    HI_CHAR *gesture_name = NULL;
    HI_ASSERT(g_osdsGesture);

    switch (resBuf.num) {
        case 0u:
            gesture_name = "wuqquexian";
            usbUartSendRead(uart_fd,zou);
            break;
        case 3u:
            gesture_name = "quexian";
            break;
            usbUartSendRead(uart_fd,ting);
        default:
            gesture_name = "unknow";
            usbUartSendRead(uart_fd,UNKNOWN);
            break;
    }

    uint32_t score = (resBuf.score) * HI_PER_BASE / SCORE_MAX;
	int res = snprintf_s(osdTxt, sizeof(osdTxt), sizeof(osdTxt) - 1, "%d_%s,%d %%", resBuf.num, gesture_name, score);
	HI_ASSERT(res > 0);

	int osdId = OsdsCreateRgn(g_osdsGesture);
	HI_ASSERT(osdId >= 0);

    int x = box.xmin / HI_OVEN_BASE * HI_OVEN_BASE;
    int y = (box.ymin - 30) / HI_OVEN_BASE * HI_OVEN_BASE; // 30: empirical value
    if (y < 0) {
        LOGD("osd_y < 0, y=%d\n", y);
        OsdsDestroyRgn(g_osdsGesture, osdId);
    } else {
        TxtRgnInit(&osdRgn, osdTxt, x, y, color, OSD_FONT_WIDTH, OSD_FONT_HEIGHT);
        OsdsSetRgn(g_osdsGesture, osdId, &osdRgn);
    }
}

/**
    将计算结果打包为resJson.
*/
static HI_CHAR* CnnGestureClassifyToJson(const RecogNumInfo items[], HI_S32 itemNum)
{
    HI_S32 jsonSize = TINY_BUF_SIZE + itemNum * TINY_BUF_SIZE; // 每个item的打包size为TINY_BUF_SIZE
    HI_CHAR *jsonBuf = (HI_CHAR*)malloc(jsonSize);
    HI_ASSERT(jsonBuf);
    HI_S32 offset = 0;

    offset += snprintf_s(jsonBuf + offset, jsonSize - offset, jsonSize - offset - 1, "[");
    for (HI_S32 i = 0; i < itemNum; i++) {
        const RecogNumInfo *item = &items[i];
        uint32_t score = item->score * HI_PER_BASE / SCORE_MAX;
        if (score < GESTURE_MIN) {
            break;
        }

        offset += snprintf_s(jsonBuf + offset, jsonSize - offset, jsonSize - offset - 1,
            "%s{ \"classify num\": %u, \"score\": %u }", (i == 0 ? "\n  " : ", "), (uint)item->num, (uint)score);
        HI_ASSERT(offset < jsonSize);
    }
    offset += snprintf_s(jsonBuf + offset, jsonSize - offset, jsonSize - offset - 1, "]");
    HI_ASSERT(offset < jsonSize);
    return jsonBuf;
}


static HI_S32 Yolo2HandDetectResnetClassifyCal(uintptr_t model,
    VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm, HI_CHAR** resJson)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    IVE_IMAGE_S img;
    DetectObjInfo objs[DETECT_OBJ_MAX] = {0};
    RectBox boxs[DETECT_OBJ_MAX] = {0};
    RectBox objBoxs[DETECT_OBJ_MAX] = {0};
    RectBox remainingBoxs[DETECT_OBJ_MAX] = {0};
    RectBox cnnBoxs[DETECT_OBJ_MAX] = {0}; // Store the results of the classification network
    RecogNumInfo numInfo[RET_NUM_MAX] = {0};
    HI_S32 resLen = 0;
    int objNum;
    int ret;
    int biggestBoxIndex;
    int num = 0;

    OsdsClear(g_osdsGesture);
    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    HI_EXP_RET(ret != HI_SUCCESS, ret, "hand_detect_cal FAIL, for YUV Frm to Img FAIL, ret=%#x\n", ret);

    objNum = HandDetectCal(&img, objs); // Send IMG to the detection net for reasoning
    for (int i = 0; i < objNum; i++) {
        cnnBoxs[i] = objs[i].box;
        RectBox *box = &objs[i].box;
        RectBoxTran(box, FRM_WIDTH, FRM_HEIGHT,
            dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        LOGI("yolo2_out: {%d, %d, %d, %d}\n",
            box->xmin, box->ymin, box->xmax, box->ymax);
        boxs[i] = *box;
    }
    biggestBoxIndex = GetBiggestHandIndex(boxs, objNum);
    LOGI("biggestBoxIndex:%d, objNum:%d\n", biggestBoxIndex, objNum);

    // When an object is detected, a rectangle is drawn in the DSTFRM
    if (biggestBoxIndex >= 0) {
        objBoxs[0] = boxs[biggestBoxIndex];
        MppFrmDrawRects(dstFrm, objBoxs, 1, RGB888_GREEN, DRAW_RETC_THICK); // Target hand objnum is equal to 1

        for (int j = 0; (j < objNum) && (objNum > 1); j++) {
            if (j != biggestBoxIndex) {
                remainingBoxs[num++] = boxs[j];
                // others hand objnum is equal to objnum -1
                MppFrmDrawRects(dstFrm, remainingBoxs, objNum - 1, RGB888_RED, DRAW_RETC_THICK);
            }
        }

        IVE_IMAGE_S imgIn;
        IVE_IMAGE_S imgDst;
        VIDEO_FRAME_INFO_S frmIn;
        VIDEO_FRAME_INFO_S frmDst;

        ret = ImgYuvCrop(&img, &imgIn, &cnnBoxs[biggestBoxIndex]); // Crop the image to classification network
        HI_EXP_LOGE(ret < 0, "ImgYuvCrop FAIL, ret = %d\n", ret);

        if ((imgIn.u32Width >= WIDTH_LIMIT) && (imgIn.u32Height >= HEIGHT_LIMIT)) {
            COMPRESS_MODE_E enCompressMode = srcFrm->stVFrame.enCompressMode;
			ret = OrigImgToFrm(&imgIn, &frmIn);
            frmIn.stVFrame.enCompressMode = enCompressMode;
			LOGI("crop u32Width = %d, img.u32Height = %d\n", imgIn.u32Width, imgIn.u32Height);
            ret = MppFrmResize(&frmIn, &frmDst, IMAGE_WIDTH, IMAGE_HEIGHT);
            ret = FrmToOrigImg(&frmDst, &imgDst);

            ret = CnnCalU8c1Img(self,  &imgDst, numInfo, HI_ARRAY_SIZE(numInfo), &resLen);
			HI_EXP_LOGE(ret < 0, "CnnCalU8c1Img FAIL, ret = %d\n", ret);
			HI_ASSERT(resLen <= sizeof(numInfo) / sizeof(numInfo[0]));
            RectBoxTran(&cnnBoxs[biggestBoxIndex], FRM_WIDTH, FRM_HEIGHT,
                dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
            HandDetectAddTxt(cnnBoxs[biggestBoxIndex], numInfo[0], ARGB1555_WHITE);

            MppFrmDestroy(&frmDst);
        }
        IveImgDestroy(&img);
        IveImgDestroy(&imgIn);
    }

    HI_CHAR *jsonBuf = CnnGestureClassifyToJson(numInfo, resLen);
    *resJson = jsonBuf;

    return ret;
}

static const AiPlug G_HAND_CLASSIFY_ITF = {
    .Prof = Yolo2HandDetectResnetClassifyProf,
    .Load = Yolo2HandDetectResnetClassifyLoad,
    .Unload = Yolo2HandDetectResnetClassifyUnload,
    .Cal = Yolo2HandDetectResnetClassifyCal,
};

const AiPlug* AiPlugItf(uint32_t* magic)
{
    if (magic) {
        *magic = AI_PLUG_MAGIC;
    }

    return (AiPlug*)&G_HAND_CLASSIFY_ITF;
}
#缺陷识别
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "sample_comm_nnie.h"
#include "nnie_sample_plug.h"

#include "hi_ext_util.h"
#include "mpp_help.h"
#include "ai_plug.h"

#define PLUG_UUID        
#define PLUG_DESC              // UTF8 encode

#define FRM_WIDTH          224
#define FRM_HEIGHT         224
// resnet_inst.wk基于开源模型resnet18重训，通过.caffemodel转wk的结果
#define MODEL_FILE_GESTURE    // 开源模型转换

#define RET_NUM_MAX         4       // 返回number的最大数目trr
#define SCORE_MAX           4096    // 最大概率对应的score
#define THRESH_MIN          30      // 可接受的概率阈值(超过此值则返回给app)

#define TXT_BEGX           20
#define TXT_BEGY           20
#define FONT_WIDTH         32
#define FONT_HEIGHT        40

static OsdSet* g_osdsGesture = NULL;
static HI_S32 g_osd0Gesture = -1;

static const HI_CHAR CNN_HAND_GESTURE[] = "{"
    "\"uuid\": " PLUG_UUID ","
    "\"desc\": " PLUG_DESC ","
    "\"frmWidth\": " HI_TO_STR(FRM_WIDTH) ","
    "\"frmHeight\": " HI_TO_STR(FRM_HEIGHT) ","
    "\"butt\": 0"
"}";

static const HI_CHAR* CnnHandGestureProf(void)
{
    return CNN_HAND_GESTURE;
}

static HI_S32 CnnHandGestureLoad(uintptr_t* model, OsdSet* osds)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    g_osdsGesture = osds;
    HI_ASSERT(g_osdsGesture);
    g_osd0Gesture = OsdsCreateRgn(g_osdsGesture);
    HI_ASSERT(g_osd0Gesture >= 0);

    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;

    return ret;
}

static HI_S32 CnnHandGestureUnload(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    OsdsClear(g_osdsGesture);

    return HI_SUCCESS;
}

/**
    将计算结果打包为resJson.
*/
HI_CHAR* CnnHandGestureToJson(const RecogNumInfo items[], HI_S32 itemNum)
{
    HI_S32 jsonSize = TINY_BUF_SIZE + itemNum * TINY_BUF_SIZE; // 每个item的打包size为TINY_BUF_SIZE
    HI_CHAR *jsonBuf = (HI_CHAR*)malloc(jsonSize);
    HI_ASSERT(jsonBuf);
    HI_S32 offset = 0;

    offset += snprintf_s(jsonBuf + offset, jsonSize - offset, jsonSize - offset - 1, "[");
    for (HI_S32 i = 0; i < itemNum; i++) {
        const RecogNumInfo *item = &items[i];
        uint32_t score = item->score * HI_PER_BASE / SCORE_MAX;
        if (score < THRESH_MIN) {
            break;
        }

        offset += snprintf_s(jsonBuf + offset, jsonSize - offset, jsonSize - offset - 1,
            "%s{ \"classify num\": %u, \"score\": %u }", (i == 0 ? "\n  " : ", "), (uint)item->num, (uint)score);
        HI_ASSERT(offset < jsonSize);
    }
    offset += snprintf_s(jsonBuf + offset, jsonSize - offset, jsonSize - offset - 1, "]");
    HI_ASSERT(offset < jsonSize);
    return jsonBuf;
}

/**
    将计算结果打包为OSD显示内容.
*/
static HI_S32 CnnHandGestureToOsd(const RecogNumInfo items[], HI_S32 itemNum, HI_CHAR* buf, HI_S32 size)
{
    HI_S32 offset = 0;
    HI_CHAR *gesture_name = NULL;

    offset += snprintf_s(buf + offset, size - offset, size - offset - 1, "hand gesture: {");
    for (HI_S32 i = 0; i < itemNum; i++) {
        const RecogNumInfo *item = &items[i];
        uint32_t score = item->score * HI_PER_BASE / SCORE_MAX;
        if (score < THRESH_MIN) {
            break;
        }
        switch (item->num) {
            case 0u:
                gesture_name = "quexian1";
                break;
            case 1u:
                gesture_name = "quexian2";
                break;
            case 2u:
                gesture_name = "wuquexian";
                break;
            default:
                gesture_name = "Unkown";
                break;
        }

        offset += snprintf_s(buf + offset, size - offset, size - offset - 1,
            "%s%s %u:%u%%", (i == 0 ? " " : ", "), gesture_name, (int)item->num, (int)score);
        HI_ASSERT(offset < size);
    }
    offset += snprintf_s(buf + offset, size - offset, size - offset - 1, " }");
    HI_ASSERT(offset < size);
    return offset;
}

static HI_S32 CnnHandGestureCal(uintptr_t model,
    VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *resFrm, HI_CHAR** resJson)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model; // reference to SDK sample_comm_nnie.h Line 99
    IVE_IMAGE_S img; // referece to SDK hi_comm_ive.h Line 143
    static HI_CHAR prevOsd[NORM_BUF_SIZE] = ""; // 安全，插件架构约定同时只会有一个线程访问插件
    HI_CHAR osdBuf[NORM_BUF_SIZE] = "";
    /*
        01-palm          02_first
        03_others
    */
    RecogNumInfo resBuf[RET_NUM_MAX] = {0};
    HI_S32 reslen = 0;
    HI_S32 ret;

    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    HI_EXP_RET(ret != HI_SUCCESS, ret, "CnnTrashClassifyCal FAIL, for YUV2RGB FAIL, ret=%#x\n", ret);

    ret = CnnCalU8c1Img(self, &img, resBuf, HI_ARRAY_SIZE(resBuf), &reslen); // 沿用该推理逻辑
    HI_EXP_LOGE(ret < 0, "cnn cal FAIL, ret=%d\n", ret);
    HI_ASSERT(reslen <= sizeof(resBuf) / sizeof(resBuf[0]));

    // 生成resJson和resOsd
    HI_CHAR *jsonBuf = CnnHandGestureToJson(resBuf, reslen);
    *resJson = jsonBuf;
    CnnHandGestureToOsd(resBuf, reslen, osdBuf, sizeof(osdBuf));

    // 仅当resJson与此前计算发生变化时,才重新打OSD输出文字
    if (strcmp(osdBuf, prevOsd) != 0) {
        HiStrxfrm(prevOsd, osdBuf, sizeof(prevOsd));

        // 叠加图形到resFrm中
        HI_OSD_ATTR_S rgn;
        TxtRgnInit(&rgn, osdBuf, TXT_BEGX, TXT_BEGY, ARGB1555_YELLOW2, FONT_WIDTH, FONT_HEIGHT);
        OsdsSetRgn(g_osdsGesture, g_osd0Gesture, &rgn);
        LOGI("CNN hand gesture: %s\n", osdBuf);
    }
    return ret;
}

static const AiPlug G_HAND_GESTURE_ITF = {
    .Prof = CnnHandGestureProf,
    .Load = CnnHandGestureLoad,
    .Unload = CnnHandGestureUnload,
    .Cal = CnnHandGestureCal,
};

const AiPlug* AiPlugItf(uint32_t* magic)
{
    if (magic) {
        *magic = AI_PLUG_MAGIC;
    }

    return (AiPlug*)&G_HAND_GESTURE_ITF;
}
