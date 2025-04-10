from flask import Flask, session, request, render_template, flash, redirect, url_for, jsonify
import time
import cv2
from script.csv_process import read_data_csv
from script.fast_alpr_script import fast_alpr_process, check_untrained_data, isMarked, check_low_confidence_data, is_start_with_numeric
from script.image_preprocessing import change_image_orientation_to_verical, crop_and_save_image, numpy_to_base64
from script.sql_db import edit_tamu_sql, list_ekspedisi_sql, list_ga_sql, all_data_sql, list_tamu, get_ekspedisi, proses_masuk_sql, proses_keluar_sql, get_kendaraan_ga
import re

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

fast_alpr = None
data = None
label = None # Hasil dari ocr plat nomor
date_ = None
time_ = None
confidence = None
ocr_result = None

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
            message = 'Jenis Kendaraan Tidak Diketahui. Input Ekspedisi, Tamu atau GA. 0x1'
            message_type = 'danger'
            return render_template('ocr.html', message=message, message_type=message_type)
        
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
                time_for_name = time.strftime("%H-%M-%S", time.localtime())
                date_ = time.strftime("%Y-%m-%d", time.localtime())
                
                image = change_image_orientation_to_verical(image, action, time_str)
                
                fast_alpr = fast_alpr_process(image)
                print(fast_alpr)
                
                if fast_alpr == False:
                    return render_template('ocr.html', message='Tidak Terdeteksi Plat Nomor atau Plat Nomor Lebih dari 1. Mohon Untuk Input Ulang.', message_type='danger')
                
                if is_start_with_numeric(fast_alpr[5]):
                    return render_template('ocr.html', message='Hasil Pembacaan Salah, Mohon Untuk Input Ulang.', message_type='danger', label=fast_alpr[5])
                
                # Dikarenakan model hanya mampu memproses maksimal 8 digit, jika ada yang lebih dari atau sama maka akan dicek ulang
                confidence = fast_alpr[0]
                ocr_result = fast_alpr[5]
                
                print("Data Asli")
                print(f"OCR: {confidence}")
                print(f"CNF: {ocr_result}")
                                
                # Kode dibawah memungkinkan memproses data jika plat nomor tidak bisa diproses OCR
                label_ = check_untrained_data(ocr_result)
                print("Data Terproses 'check_untrained_data'")
                print(f"OCR: {label_}")
                print(f"CNF: {confidence}")
            
                # Cek plat nomor jika nilai confidence rendah
                label, status_checking = check_low_confidence_data(label_, confidence)
                print("Data Terproses 'check_low_confidence_data'")
                print(f"OCR: {label}")
                print(f"CNF: {confidence}")
                
                # Jika plat ada di check_untrained_data atau check_low_confidence_data maka tidak perlu proses Marked
                if status_checking:
                    # confidence 0.99 tidak mungkin ada, maks 0.9
                    # Jika ada 0.99 confidence maka itu sudah diproses pengecekan
                    confidence = 0.99
                    
                if isMarked(label, confidence):
                    marked = "MARKED"
                    print(marked)
                else:
                    marked = ""
                    print("NOT MARKED")
                
                # image, x1, y1, x2, y2, action, ocr_result, confidence, datetime, date, time, marker
                save = crop_and_save_image(image, fast_alpr[1], fast_alpr[2], fast_alpr[3], fast_alpr[4], action, label, fast_alpr[0], time_str, date_, time_for_name, marked)
                
                img_temp = cv2.imread(save[0])
                data = numpy_to_base64(img_temp)
                
                if save is None:
                    return render_template('ocr.html', message='Proses Menyimpan Data Gagal.', message_type='danger')
                
                # return render_template('ocr.html', message=fast_alpr, data=data, message_type='success')
            
            except (IOError, SyntaxError, TypeError):
                return render_template('ocr.html', message='Gagal Memproses Gambar. Mohon Untuk Input Ulang.', message_type='danger')
        
        """
        Algoritma proses Masuk/Keluar
        1. Masuk
           - Cek Jenis Kendaraan
             - Eskpedisi
               - Cek Data Ekspedisi
               - Proses Masuk, Jika sudah ada data masuk gagalkan, return OKE
               - Jika belum terdaftar return waring, jika sudah ada gagalkan return waring
             - GA
               - Cek apakah benar kendaraan GA, jika benar lanjut, jika gagal return error
               - Proses Masuk, Jika sudah ada gagalkan return OKE
             - Tamu
               - Proses Masuk, Jika sudah ada gagalkan return OKE
             - Else
               - return danger
               
        2. Keluar
           - Proses Keluar
           - Jika sudah ada data keluar gagalkan, return OKE
        """
        
        # Add for git
              
        if action == 'Masuk':
            last_loc = None
            if entryType == 'Ekspedisi':
                ekspedisi = get_ekspedisi(label)
                
                if ekspedisi is not None:
                    last_loc = proses_masuk_sql(date_, label, time_, 'security', ekspedisi[0])
                    
                    if last_loc:
                        message = 'Ekspedisi: {} Sudah Didalam. Tidak Bisa Diproses Masuk 2x.'.format(label)
                        message_type = 'danger'
                        return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                    else:
                        message = 'Ekspedisi: {} Berhasil Masuk.'.format(label)
                        message_type = 'success'
                        return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label, type=ekspedisi)
                else:
                    last_loc = proses_masuk_sql(date_, label, time_, 'security', entryType)
                    
                    if last_loc:
                        message = 'Ekspedisi: {} Sudah Didalam. Tidak Bisa Diproses Masuk 2x.'.format(label)
                        message_type = 'danger'
                        return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                    else:
                        message = 'Ekspedisi: {} Tidak Terdaftar. Proses Tetap Dilanjutkan.'.format(label)
                        message_type = 'warning'
                        return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                
            elif entryType == 'GA':
                status_kendaraan_ga = get_kendaraan_ga(label)
                
                if status_kendaraan_ga:
                    last_loc = proses_masuk_sql(date_, label, time_, 'security', entryType, km)
                    if last_loc:
                        message = 'GA: {} Sudah Didalam. Tidak Bisa Diproses Masuk 2x.'.format(label)
                        message_type = 'danger'
                        return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                    else:
                        message = 'GA: {} Berhasil Masuk.'.format(label)
                        message_type = 'success'
                        return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                else:
                    message = '{} Tidak Terdaftar Sebagai GA.'.format(label)
                    message_type = 'danger'
                    return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                
            elif entryType == 'Tamu':
                last_loc = proses_masuk_sql(date_, label, time_, 'security', entryType)
                if last_loc:
                    message = 'Tamu: {} Sudah Didalam. Tidak Bisa Diproses Masuk 2x.'.format(label)
                    message_type = 'danger'
                    return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                else:
                    message = 'Tamu: {} Berhasil Masuk. Hubungi Admin Untuk Pengisian Data PIC dan Keperluan.'.format(label)
                    message_type = 'success'
                    return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
            
            else:
                message = 'Jenis Kendaraan Tidak Diketahui. Input Ekspedisi, Tamu atau GA.'
                message_type = 'danger'
                return render_template('ocr.html', message=message, message_type=message_type)
            
        elif action == 'Keluar':
            last_loc = proses_keluar_sql(date_, label, time_, 'security', km)
            if last_loc:
                message = 'Kendaraan: {} Sudah Diluar. Tidak Bisa Diproses Keluar 2x.'.format(label)
                message_type = 'danger'
                return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
            else:
                message = 'Kendaraan: {} Berhasil Keluar.'.format(label)
                message_type = 'success'
                return render_template('ocr.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
         
        else:
            message = 'Tujuan Tidak Diketahui. Input Masuk atau Keluar'
            message_type = 'danger'
            return render_template('ocr.html', message=message, message_type=message_type)
            
    return render_template('ocr.html')
    
@app.route("/ocr/get_data_all_ocr")
def get_data_all_ocr():
    data = read_data_csv()
    return jsonify(data)

STRING_REGEX = re.compile(r"^[A-Za-z\s]+$")
@app.route("/ocr/edit_tamu", methods=['GET', 'POST'])
def edit_tamu():
    if not session.get('authenticated'):
        return redirect(url_for('login_ocr'))
    
    list_keperluan = ['Interview', 'BS', 'Sampah', 'Tamu', 'Sales', 'Lainnya']
    data_tamu = list_tamu()
    
    if request.method == 'POST':
        try:
            no_mobil = request.form['noMobil']
            pic_stt = request.form['picSTT']
            keperluan = request.form['keperluan']
            datetime = request.form['dateTime']
        except:
            print("Data tidak ada")
            
        print(f"{no_mobil}, {pic_stt}, {keperluan}, {datetime}")
            
        if not STRING_REGEX.match(pic_stt):
            flash('Input PIC STT harus berupa huruf.', 'danger')
            return redirect(url_for('edit_tamu'))
            
        status_edit = edit_tamu_sql(datetime, no_mobil, keperluan, pic_stt)
        if status_edit != False:
            flash(f'Data Tamu {no_mobil} Berhasil Diperbarui.', 'success')
            return redirect(url_for('edit_tamu'))
    
    return render_template('edit_tamu.html', list_tamu=data_tamu, list_keperluan=list_keperluan)

@app.route("/ocr/list_ga")
def list_ga():
    data = list_ga_sql()
    return render_template('list_ga.html', data = data)

@app.route("/ocr/list_ekspedisi")
def list_ekspedisi():
    data = list_ekspedisi_sql()
    return render_template('list_ekspedisi.html', data = data)

@app.route("/ocr/data_ocr")
def data_ocr():
    data = all_data_sql()
    return render_template('data_ocr.html', data = data)

@app.route("/ocr/save_img", methods=['GET', 'POST'])
def save_img():
    # if not session.get('authenticated'):
    #     return redirect(url_for('login_ocr'))
    
    if request.method == 'POST':
        action = request.form['action']
        image = request.files['image']
        
        try:
            entryType = request.form['entryType']
        except:
            message = 'Jenis Kendaraan Tidak Diketahui. Input Ekspedisi, Tamu atau GA. 0x1'
            message_type = 'danger'
            return render_template('save-img.html', message=message, message_type=message_type)
        
        if action == False or image == False or entryType == False:
            return render_template('save-img.html', message='Form Tidak Lengkap. Mohon Untuk Input Ulang.', message_type='danger')  

        km = None
        
        if entryType == 'GA':
            km = request.form.get('km')
            if km == '' or km is None:
                return render_template('save-img.html', message='KM Kendaraan GA Tidak Valid. Mohon Untuk Input Ulang.', message_type='danger')

        print(f"Action: {action}, Img: {type(image)}, Entry Type: {entryType}, KM: {km}")
        
        if image:
            try:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                time_ = time.strftime("%H:%M:%S", time.localtime())
                time_for_name = time.strftime("%H-%M-%S", time.localtime())
                date_ = time.strftime("%Y-%m-%d", time.localtime())
                
                image = change_image_orientation_to_verical(image, action, time_str)
                
                fast_alpr = fast_alpr_process(image)
                print(fast_alpr)
                
                if fast_alpr == False:
                    return render_template('save-img.html', message='Tidak Terdeteksi Plat Nomor atau Plat Nomor Lebih dari 1. Mohon Untuk Input Ulang.', message_type='danger')
                
                if is_start_with_numeric(fast_alpr[5]):
                    return render_template('save-img.html', message='Hasil Pembacaan Salah, Mohon Untuk Input Ulang.', message_type='danger', label=fast_alpr[5])
                
                # Dikarenakan model hanya mampu memproses maksimal 8 digit, jika ada yang lebih dari atau sama maka akan dicek ulang
                confidence = fast_alpr[0]
                ocr_result = fast_alpr[5]
                
                # Jika confidence kurang dari 0.7, return false
                if confidence < 0.7:
                    return render_template('save-img.html', message='Hasil Pembacaan Rendah, Mohon Untuk Input Ulang.', message_type='danger', label=fast_alpr[5], confidence=confidence)
                
                print("Data Asli")
                print(f"OCR: {confidence}")
                print(f"CNF: {ocr_result}")
                                
                # Kode dibawah memungkinkan memproses data jika plat nomor tidak bisa diproses OCR
                label_ = check_untrained_data(ocr_result)
                print("Data Terproses 'check_untrained_data'")
                print(f"OCR: {label_}")
                print(f"CNF: {confidence}")
            
                # Cek plat nomor jika nilai confidence rendah
                label, status_checking = check_low_confidence_data(label_, confidence)
                print("Data Terproses 'check_low_confidence_data'")
                print(f"OCR: {label}")
                print(f"CNF: {confidence}")
                
                # Jika plat ada di check_untrained_data atau check_low_confidence_data maka tidak perlu proses Marked
                if status_checking:
                    # confidence 0.99 tidak mungkin ada, maks 0.9
                    # Jika ada 0.99 confidence maka itu sudah diproses pengecekan
                    confidence = 0.99
                    
                if isMarked(label, confidence):
                    marked = "MARKED"
                    print(marked)
                else:
                    marked = ""
                    print("NOT MARKED")
                
                # image, x1, y1, x2, y2, action, ocr_result, confidence, datetime, date, time, marker
                save = crop_and_save_image(image, fast_alpr[1], fast_alpr[2], fast_alpr[3], fast_alpr[4], action, label, fast_alpr[0], time_str, date_, time_for_name, marked)
                
                img_temp = cv2.imread(save[0])
                data = numpy_to_base64(img_temp)
                
                if save is None:
                    return render_template('save-img.html', message='Proses Menyimpan Data Gagal.', message_type='danger')
                
                # return render_template('save-img.html', message=fast_alpr, data=data, message_type='success')
            
            except (IOError, SyntaxError, TypeError):
                return render_template('save-img.html', message='Gagal Memproses Gambar. Mohon Untuk Input Ulang.', message_type='danger')
        
        """
        Algoritma proses Masuk/Keluar
        1. Masuk
           - Cek Jenis Kendaraan
             - Eskpedisi
               - Cek Data Ekspedisi
               - Proses Masuk, Jika sudah ada data masuk gagalkan, return OKE
               - Jika belum terdaftar return waring, jika sudah ada gagalkan return waring
             - GA
               - Cek apakah benar kendaraan GA, jika benar lanjut, jika gagal return error
               - Proses Masuk, Jika sudah ada gagalkan return OKE
             - Tamu
               - Proses Masuk, Jika sudah ada gagalkan return OKE
             - Else
               - return danger
               
        2. Keluar
           - Proses Keluar
           - Jika sudah ada data keluar gagalkan, return OKE
        """
        
        # Add for git
              
        if action == 'Masuk':
            last_loc = None
            if entryType == 'Ekspedisi':
                ekspedisi = get_ekspedisi(label)
                
                if ekspedisi is not None:
                    last_loc = proses_masuk_sql(date_, label, time_, 'security', ekspedisi[0])
                    
                    if last_loc:
                        message = 'Ekspedisi: {} Sudah Didalam. Tidak Bisa Diproses Masuk 2x.'.format(label)
                        message_type = 'danger'
                        return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                    else:
                        message = 'Ekspedisi: {} Berhasil Masuk.'.format(label)
                        message_type = 'success'
                        return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label, type=ekspedisi)
                else:
                    last_loc = proses_masuk_sql(date_, label, time_, 'security', entryType)
                    
                    if last_loc:
                        message = 'Ekspedisi: {} Sudah Didalam. Tidak Bisa Diproses Masuk 2x.'.format(label)
                        message_type = 'danger'
                        return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                    else:
                        message = 'Ekspedisi: {} Tidak Terdaftar. Proses Tetap Dilanjutkan.'.format(label)
                        message_type = 'warning'
                        return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                
            elif entryType == 'GA':
                status_kendaraan_ga = get_kendaraan_ga(label)
                
                if status_kendaraan_ga:
                    last_loc = proses_masuk_sql(date_, label, time_, 'security', entryType, km)
                    if last_loc:
                        message = 'GA: {} Sudah Didalam. Tidak Bisa Diproses Masuk 2x.'.format(label)
                        message_type = 'danger'
                        return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                    else:
                        message = 'GA: {} Berhasil Masuk.'.format(label)
                        message_type = 'success'
                        return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                else:
                    message = '{} Tidak Terdaftar Sebagai GA.'.format(label)
                    message_type = 'danger'
                    return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                
            elif entryType == 'Tamu':
                last_loc = proses_masuk_sql(date_, label, time_, 'security', entryType)
                if last_loc:
                    message = 'Tamu: {} Sudah Didalam. Tidak Bisa Diproses Masuk 2x.'.format(label)
                    message_type = 'danger'
                    return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
                else:
                    message = 'Tamu: {} Berhasil Masuk. Hubungi Admin Untuk Pengisian Data PIC dan Keperluan.'.format(label)
                    message_type = 'success'
                    return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
            
            else:
                message = 'Jenis Kendaraan Tidak Diketahui. Input Ekspedisi, Tamu atau GA.'
                message_type = 'danger'
                return render_template('save-img.html', message=message, message_type=message_type)
            
        elif action == 'Keluar':
            last_loc = proses_keluar_sql(date_, label, time_, 'security', km)
            if last_loc:
                message = 'Kendaraan: {} Sudah Diluar. Tidak Bisa Diproses Keluar 2x.'.format(label)
                message_type = 'danger'
                return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
            else:
                message = 'Kendaraan: {} Berhasil Keluar.'.format(label)
                message_type = 'success'
                return render_template('save-img.html', message=message, message_type=message_type, data=data, confidence=confidence, label=label)
         
        else:
            message = 'Tujuan Tidak Diketahui. Input Masuk atau Keluar'
            message_type = 'danger'
            return render_template('save-img.html', message=message, message_type=message_type)
            
    return render_template('save-img.html')
   

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )