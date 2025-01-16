from flask import Flask, session, request, render_template, flash, redirect, url_for, jsonify
import time
from script.csv_process import read_data_csv
from script.fast_alpr_script import fast_alpr_process
from script.image_preprocessing import change_image_orientation_to_verical, crop_and_save_image

app = Flask(__name__)
app.secret_key = 'itbekasioke'
USER_SECRET_KEY = 'user123'
USER_EDIT_TAMU_KEY = 'tamu123'

@app.route('/ocr/login_ocr', methods=['GET', 'POST'])
def login_ocr():
    if request.method == 'POST':
        secret_key = request.form.get('secret_key')
        if secret_key == USER_SECRET_KEY:
            session['authenticated'] = True
            return redirect(url_for('ocr'))
        elif secret_key == USER_EDIT_TAMU_KEY:
            session['authenticated'] = True
            return redirect(url_for('edit_tamu'))
        else:
            flash('Password salah, silahkan coba kembali.', 'danger')
    
    return render_template('login.html')

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if not session.get('authenticated'):
        return redirect(url_for('login_ocr'))
    
    if request.method == 'POST':
        action = request.form['action']
        image = request.files['image']
        
        try:
            entryType = request.form['entryType']
        except:
            return render_template('ocr.html', message='Jenis Kendaraan Tidak Valid. Mohon Untuk Input Ulang.', message_type='danger')
        
        if action == False or image == False or entryType == False:
            return render_template('ocr.html', message='Form Tidak Lengkap. Mohon Untuk Input Ulang.', message_type='danger')  

        km = None
        
        if entryType == 'GA':
            km = request.form.get('km')
            if km == '' or km is None:
                return render_template('ocr.html', message='KM Kendaraan GA Tidak Valid. Mohon Untuk Input Ulang.', message_type='danger')

        print(f"Action: {action}, Img: {type(image)}, Entry Type: {entryType}, KM: {km}")
        
        if image:
            try:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                time_ = time.strftime("%H:%M:%S", time.localtime())
                date_ = time.strftime("%Y-%m-%d", time.localtime())
                
                image = change_image_orientation_to_verical(image, action, time_str)
                
                fast_alpr = fast_alpr_process(image)
                print(fast_alpr)
                
                if fast_alpr == False:
                    return render_template('ocr.html', message='Tidak Terdeteksi Plat Nomor atau Plat Nomor Lebih dari 1. Mohon Untuk Input Ulang.', message_type='danger')
                
                # Dikarenakan model hanya mampu memproses maksimal 8 digit, jika ada yang lebih dari atau sama maka akan dicek ulang
                if isMarked(fast_alpr[5]):
                    marked = "MARKED"
                else:
                    marked = ""
                
                # image, x1, y1, x2, y2, action, ocr_result, confidence, datetime, date, time, marker
                save = crop_and_save_image(image, fast_alpr[1], fast_alpr[2], fast_alpr[3], fast_alpr[4], action, fast_alpr[5], fast_alpr[0], time_str, date_, time_, marked)
                
                if save is None:
                    return render_template('ocr.html', message='Proses Menyimpan Data Gagal.', message_type='danger')
                
                return render_template('ocr.html', message=fast_alpr, message_type='success')
            
            except:
                return render_template('ocr.html', message='Gagal Memproses Gambar. Mohon Untuk Input Ulang.', message_type='danger')
                
    return render_template('ocr.html')

def isMarked(plate):
    if len(plate) == 8:
        return (
            plate[:2].isalpha() and    # First two are letters
            plate[2:6].isdigit() and   # Middle 4 are digits
            plate[6:].isalpha()        # Last two are letters
        )
    
@app.route("/ocr/get_data_all_ocr")
def get_data_all_ocr():
    data = read_data_csv()
    return jsonify(data)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )