// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
// import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as mobilenetModule from "./mobilenet";
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import _ from "lodash";

// Number of classes to classify
const NUM_CLASSES = 5;
// Webcam Image size. Must be 227. 
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;


class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;
    this.responseAudios = [];
    this.lastState = -1;
    this.lastKey = '';

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');

    // Add video element to DOM
    document.body.appendChild(this.video);
    //录音
    var rec = Recorder();
    let exampleKey = ['Z', 'X', 'C', 'V', 'B'];
    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';
      div.style.display = 'flex';
      div.style.alignItems = 'center';

      // Listen for key events
      document.body.addEventListener('keydown', (e) => {
        let _self = this;
        const key = _.upperCase(e.key);
        let exampleNum = exampleKey.indexOf(key);
        this.training = exampleNum;
        if (key != this.lastKey) {
          rec.open(function () { //打开麦克风授权获得相关资源
              rec.start(); //开始录音
              setTimeout(function () {
                rec.stop(function (blob, duration) { //到达指定条件停止录音，拿到blob对象想干嘛就干嘛：立即播放、上传
                  var blobUrl = URL.createObjectURL(blob);
                  console.log(blobUrl, "时长:" + duration + "ms");
                  rec.close(); //释放录音资源
                  _self.responseAudios[exampleNum].src = blobUrl;
                }, function (msg) {
                  console.log("录音失败:" + msg);
                });
              }, 5000);
            },
            function (msg) { //未授权或不支持
              console.log("无法录音:" + msg);
            });
        }
        this.lastKey = key;
      });
      document.body.addEventListener('keyup', (e) => {
        this.training = -1;
        const key = _.upperCase(e.key);
        this.lastKey = key;
      });

      // Create info text
      const infoText = document.createElement('span')
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);

      //播放声音
      const audio = document.createElement('audio');
      audio.controls = true;
      div.appendChild(audio);
      this.responseAudios.push(audio);
    }


    // Setup webcam
    navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    this.start();
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));

    const loadToast = document.createElement('div');
    loadToast.style.height = '32px';
    loadToast.style.width = '100%';
    loadToast.style.backgroundColor = 'green';
    loadToast.style.top = 0;
    loadToast.style.left = 0;
    loadToast.style.position = 'absolute';
    loadToast.innerText = 'loaded!';
    loadToast.style.textAlign = 'center';
    loadToast.style.lineHeight = '32px';
    loadToast.style.boxShadow = '0 1px 6px rgba(0,0,0,.2)';
    loadToast.style.borderColor = '#eee';
    document.body.appendChild(loadToast);
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training)
      }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`;
          }
        }

        let obj = _.cloneDeep(res.confidences);
        let arr = [];
        for (let key in obj) {
          arr.push({
            index: key,
            value: obj[key]
          });
        }
        arr.sort((a, b) => {
          return a.value - b.value;
        });
        let len = arr.length;
        let max = arr[len - 1];

        if (this.lastState !== max.index) {
          this.responseAudios[max.index].play()
        }
        this.lastState = max.index
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}

window.addEventListener('load', () => new Main());