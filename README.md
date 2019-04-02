# tensorflow-demo
This is a small project build with tensorflow.js for someone's graduation thesis. In order to enable users in mainland China to access [MobileNet](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet) we change the source.

## Usage
To use the code, first install the Javascript dependencies by running  
```bash
npm install
```
Then start the local budo web server by running 
```bash
npm start
```
This will start a web server on [`localhost:9966`](http://localhost:9966). Try and allow permission to your webcam, and add some examples by pressing keys below.
<kbd>z</kbd> <kbd>x</kbd> <kbd>c</kbd> <kbd>v</kbd> <kbd>b</kbd>
## Notice
**The getUserMedia() method is only available in secure contexts. A secure context is one the browser is reasonably confident contains a document which was loaded securely, using HTTPS/TLS, and has limited exposure to insecure contexts.**
If you want to deploy it in your server make sure using HTTPS.