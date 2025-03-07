import mysql.connector
import re

def db_connection(host, user, password, database):
    """Connect to MySQL database.

    :param host: hostname or ip address of mysql server
    :param user: username to use for connection
    :param password: password to use for connection
    :param database: database name to use
    :return: a database connection
    """
    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

def iot_223():
    return db_connection('192.168.15.223', 'admin', 'itbekasioke', 'iot')

def parkir_220():
    return db_connection('192.168.15.220', 'user_external_220', 'Sttbekasioke123!', 'parkir')

def db_pegawai_220():
    return db_connection('192.168.15.220', 'user_external_220', 'Sttbekasioke123!', 'db_pegawai')

# Berfungsi untuk menghapus spasi dan mengubah semua huruf menjadi huruf kapital
def normalize_no_mobil(no_mobil):
    return re.sub(r'\s+', '', no_mobil).upper()

def get_ekspedisi(no_mobil):
    no_mobil_normalized = normalize_no_mobil(no_mobil)
    
    conn_parkir = parkir_220()
    cursor_parkir = conn_parkir.cursor()
    
    cursor_parkir.execute("""
        SELECT nama_ekspedisi, panjang, lebar, tinggi, cbm, jn_kendaraan FROM pengukuran
        WHERE REPLACE(REPLACE(nomor_polisi, ' ', ''), '-', '') = %s
    """, (no_mobil_normalized,))
    result = cursor_parkir.fetchone()
    
    cursor_parkir.close()
    conn_parkir.close()
    
    return result if result else None

def proses_masuk_sql(tanggal, no_mobil, jam_masuk_pabrik, user_in, ekspedisi, km = None):
    # Ganti variabel conn_masuk sesuai tujuan db
    conn_masuk = iot_223()
    cursor_masuk = conn_masuk.cursor()
    
    # Cek status terakhir kendaraan, apakah didalam atau diluar
    cursor_masuk.execute("""
        SELECT * FROM ocr
        WHERE no_mobil = %s AND (tanggal_keluar IS NULL OR jam_keluar IS NULL)
    """, (no_mobil,))
    check_last_status = cursor_masuk.fetchone()
    
    # Jika terdapat data hasil dari query diatas maka return masuk
    if check_last_status:
        return 'didalam'
    
    # Jika data tidak ada maka insert data ke db
    cursor_masuk.execute("""
        INSERT INTO ocr (tanggal, no_mobil, jam_masuk_pabrik, user_in, ekspedisi)
        VALUES (%s, %s, %s, %s, %s)
    """, (tanggal, no_mobil, jam_masuk_pabrik, user_in, ekspedisi))
    conn_masuk.commit()
    
    # Jika ada input km
    if km is not None:
        ga_km_process(no_mobil, km, 'Masuk')
    
    cursor_masuk.close()
    conn_masuk.close()
    
def proses_keluar_sql(tanggal, no_mobil, jam_keluar_pabrik, user_out, km = None):
    conn_keluar = iot_223()
    cursor_keluar = conn_keluar.cursor()
    no_mobil_normalized = normalize_no_mobil(no_mobil)
    
    # Cek Status Terakhir
    cursor_keluar.execute("""
        SELECT * FROM ocr
        WHERE no_mobil = %s AND tanggal_keluar IS NULL AND jam_masuk_pabrik IS NOT NULL
    """, (no_mobil_normalized,))
    result = cursor_keluar.fetchone()
    
    if not result:
        return True
    
    cursor_keluar.execute("""
        UPDATE ocr
        SET tanggal_keluar = %s, jam_keluar = %s, user_out = %s
        WHERE no_mobil = %s AND tanggal_keluar IS NULL
    """, (tanggal, jam_keluar_pabrik, user_out, no_mobil_normalized))
    conn_keluar.commit()
    
    if km is not None:
        ga_km_process(no_mobil, km, 'Keluar')
    
    cursor_keluar.close()
    conn_keluar.close()

def ga_km_process(no_mobil, km, action):
    conn_ga_km_process = iot_223()
    cursor_ga_km_process = conn_ga_km_process.cursor()
    no_mobil_normalized = normalize_no_mobil(no_mobil)
    
    try:
        # Step 1: Get the latest tanggal for the given no_mobil
        cursor_ga_km_process.execute("""
            SELECT MAX(tanggal) 
            FROM ocr 
            WHERE no_mobil = %s
        """, (no_mobil_normalized,))
        latest_date = cursor_ga_km_process.fetchone()[0]
        if latest_date is None:
            return False
        
        # Step 2: Get the latest time for the given no_mobil
        cursor_ga_km_process.execute("""
            SELECT MAX(jam_masuk_pabrik) 
            FROM ocr 
            WHERE no_mobil = %s
            AND tanggal = %s
        """, (no_mobil_normalized, latest_date))
        latest_time = cursor_ga_km_process.fetchone()[0]
        if latest_time is None:
            return False
        
        if action == 'Masuk':
            print("Proses km_in")
            cursor_ga_km_process.execute("""
                UPDATE ocr
                SET km_in = %s
                WHERE no_mobil = %s 
                AND ekspedisi = 'GA'
                AND tanggal = %s
                AND jam_masuk_pabrik = %s
            """, (km, no_mobil_normalized, latest_date, latest_time))
            conn_ga_km_process.commit()

        elif action == 'Keluar':
            print("Proses km_out")
            cursor_ga_km_process.execute("""
                UPDATE ocr
                SET km_out = %s
                WHERE no_mobil = %s 
                AND ekspedisi = 'GA'
                AND tanggal = %s
                AND jam_masuk_pabrik = %s
            """, (km, no_mobil_normalized, latest_date, latest_time))
            conn_ga_km_process.commit()

    except Exception as e:
        print(f"An error occurred: {e}")
        conn_ga_km_process.rollback()  # Rollback in case of error
        
    cursor_ga_km_process.close()
    conn_ga_km_process.close()
    
def get_kendaraan_ga(no_mobil):
    no_mobil_normalized = normalize_no_mobil(no_mobil)
    
    conn_db_pegawai = db_pegawai_220()
    cursor_db_pegawai = conn_db_pegawai.cursor()
    
    cursor_db_pegawai.execute("""
        SELECT jenis_kendaraan FROM kendaraan
        WHERE REPLACE(REPLACE(nopol, ' ', ''), '-', '') = %s
    """, (no_mobil_normalized,))
    result = cursor_db_pegawai.fetchone()
    
    cursor_db_pegawai.close()
    conn_db_pegawai.close()
    
    return True if result else False

def edit_tamu_sql(datetime, no_mobil, keperluan, pic_stt):
    conn_edit_tamu = iot_223()
    cursor_edit_tamu = conn_edit_tamu.cursor()
    cursor_edit_tamu.execute("""
        UPDATE ocr
        SET pic_stt = %s, keperluan = %s
        WHERE no_mobil = %s
        AND ekspedisi = 'Tamu'
        AND CONCAT(tanggal, ' ', jam_masuk_pabrik) = %s
    """, (pic_stt, keperluan, no_mobil, datetime))
    conn_edit_tamu.commit()
    
    cursor_edit_tamu.close()
    conn_edit_tamu.close()

# Melihat daftar tamu
def list_tamu():
    # SELECT * FROM `ocr` WHERE ekspedisi = 'Tamu' AND pic_stt IS NULL AND keperluan IS NULL ORDER BY tanggal DESC, jam_masuk_pabrik DESC;
    
    conn_edit_tamu = iot_223()
    cursor_edit_tamu = conn_edit_tamu.cursor()
    cursor_edit_tamu.execute("""
        SELECT 
            CONCAT(tanggal, ' ', jam_masuk_pabrik) AS waktu_masuk,
            no_mobil,
            ekspedisi,
            pic_stt,
            keperluan,
            CONCAT(tanggal_keluar, ' ', jam_keluar) AS waktu_keluar
        FROM 
            ocr
        WHERE ekspedisi = 'Tamu'
        -- AND pic_stt IS NULL
        -- AND keperluan IS NULL
        ORDER BY tanggal DESC, jam_masuk_pabrik DESC
    """)
    result = cursor_edit_tamu.fetchall()
    
    cursor_edit_tamu.close()
    conn_edit_tamu.close()
    return result

def list_ekspedisi_sql():
    conn_list_ekspedisi = iot_223()
    cursor_list_ekspedisi = conn_list_ekspedisi.cursor()
    cursor_list_ekspedisi.execute("""
        SELECT 
            CONCAT(tanggal, ' ', jam_masuk_pabrik) AS waktu_masuk,
            no_mobil,
            ekspedisi,
            CONCAT(tanggal_keluar, ' ', jam_keluar) AS waktu_keluar
        FROM 
            ocr
        WHERE ekspedisi != 'Tamu'
        AND ekspedisi != 'GA'
        ORDER BY tanggal DESC, jam_masuk_pabrik DESC
    """)
    result = cursor_list_ekspedisi.fetchall()
    
    cursor_list_ekspedisi.close()
    conn_list_ekspedisi.close()
    return result

def list_ga_sql():
    conn_list_ga = iot_223()
    cursor_list_ga = conn_list_ga.cursor()
    cursor_list_ga.execute("""
        SELECT 
            CONCAT(tanggal, ' ', jam_masuk_pabrik) AS waktu_masuk,
            no_mobil,
            km_in,
            km_out,
            CONCAT(tanggal_keluar, ' ', jam_keluar) AS waktu_keluar
        FROM 
            ocr
        WHERE ekspedisi = 'GA'
        ORDER BY tanggal DESC, jam_masuk_pabrik DESC
    """)
    result = cursor_list_ga.fetchall()
    
    cursor_list_ga.close()
    conn_list_ga.close()
    return result

def all_data_sql():
    conn_all_data = iot_223()
    cursor_all_data = conn_all_data.cursor()
    cursor_all_data.execute("""
        SELECT 
            CONCAT(tanggal, ' ', jam_masuk_pabrik) AS waktu_masuk,
            no_mobil,
            ekspedisi,
            pic_stt,
            keperluan,
            CONCAT(km_in, ' | ', km_out) AS km,
            CONCAT(tanggal_keluar, ' ', jam_keluar) AS Waktu_keluar
        FROM 
            ocr
        ORDER BY tanggal DESC, jam_masuk_pabrik DESC
    """)
    result = cursor_all_data.fetchall()
    
    cursor_all_data.close()
    conn_all_data.close()
    return result