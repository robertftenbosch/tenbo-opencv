from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cartoon')
def cartoon():
    return render_template('cartoon.html')
@app.route('/pencil')
def pencil():
    return render_template('pencil.html')
@app.route('/sketch')
def sketch():
    return render_template('sketch.html')
def gen(camera, is_cartoon=False, is_pencil=False,is_sketch=False):
    while True:
        if is_cartoon:
            frame = camera.get_cartoon_frame()
            yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        elif is_pencil:
            frame = camera.get_pencil_frame()
            yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        elif is_sketch:
            frame = camera.get_sketch_frame()
            yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            frame = camera.get_frame()
            yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_cartoon')
def video_feed_cartoon():
    return Response(gen(VideoCamera(), is_cartoon=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_pencil')
def video_feed_pencil():
    return Response(gen(VideoCamera(), is_pencil=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_sketch')
def video_feed_sketch():
    return Response(gen(VideoCamera(), is_sketch=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)