import streamlit as st
import pandas as pd
import requests

# URL endpoint API Django
API_URL = 'http://localhost:8000/api/predict/'

def get_prediction(features):
    """Kirim permintaan POST ke endpoint API dan kembalikan hasilnya."""
    response = requests.post(API_URL, json=features)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Terjadi kesalahan saat mengambil prediksi.")
        return None

def main():
    st.title("Prediksi Kelayakan Agunan")

    # Input dari pengguna
    tipe_agunan = st.selectbox('Tipe Agunan', ['Tak berwujud', 'Berwujud'])
    kondisi_agunan = st.selectbox('Kondisi Agunan', ['Resiko Tinggi', 'Sangat Lemah', 'Lemah', 'Meragukan', 'Cukup', 'Kuat', 'Sangat Kuat'])
    kemudahan_dijual_kembali = st.selectbox('Kemudahan Dijual Kembali', ['Resiko Tinggi', 'Sangat Lemah', 'Lemah', 'Meragukan', 'Cukup', 'Kuat', 'Sangat Kuat'])
    status_lunas_agunan = st.selectbox('Status Lunas Agunan', ['Belum Lunas', 'Lunas'])
    agunan_dibanyak_pinjaman = st.selectbox('Agunan Dibanyak Pinjaman', ['Tidak', 'Ya'])
    kesediaan_untuk_digadaikan = st.selectbox('Kesediaan Untuk Digadaikan', ['Tidak', 'Ya'])
    jenis_asset_bergerak_atau_tidak = st.selectbox('Jenis Asset Bergerak Atau Tidak', ['Tidak', 'Ya'])
    riwayat_agunan_sebelumnya = st.selectbox('Riwayat Agunan Sebelumnya', ['Tidak Pernah', 'Pernah'])
    status_kepemilikan_asset = st.selectbox('Status Kepemilikan Asset', ['Non-Pribadi', 'Pribadi'])
    asuransi_agunan = st.selectbox('Asuransi Agunan', [False, True])
    pekerjaan = st.text_input('Pekerjaan')
    tujuan_pinjaman = st.text_input('Tujuan Pinjaman')
    agunan = st.text_input('Agunan')
    merk_aset = st.text_input('Merk Aset')
    status_agunan_terburuk = st.number_input('Status Agunan Terburuk', min_value=0)
    jumlah_agunan = st.number_input('Jumlah Agunan', min_value=0)
    nilai_agunan = st.number_input('Nilai Agunan', min_value=0)
    tenor = st.number_input('Tenor', min_value=0)
    usia_agunan = st.number_input('Usia Agunan', min_value=0)
    aset_maya = st.number_input('Asumsi Aset Maya', min_value=0)
    kenaikan_nilai_property_agunan = st.number_input('Kenaikan Nilai Property Agunan', min_value=0)
    surat_berharga = st.number_input('Surat Berharga', min_value=0)
    
    features = {
        'tipe_agunan': tipe_agunan,
        'kondisi_agunan': kondisi_agunan,
        'kemudahan_dijual_kembali': kemudahan_dijual_kembali,
        'status_lunas_agunan': status_lunas_agunan,
        'agunan_dibanyak_pinjaman': agunan_dibanyak_pinjaman,
        'kesediaan_untuk_digadaikan': kesediaan_untuk_digadaikan,
        'jenis_asset_bergerak_atau_tidak': jenis_asset_bergerak_atau_tidak,
        'riwayat_agunan_sebelumnya': riwayat_agunan_sebelumnya,
        'status_kepemilikan_asset': status_kepemilikan_asset,
        'asuransi_agunan': asuransi_agunan,
        'pekerjaan': pekerjaan,
        'tujuan_pinjaman': tujuan_pinjaman,
        'agunan': agunan,
        'merk_aset': merk_aset,
        'status_agunan_terburuk': status_agunan_terburuk,
        'jumlah_agunan': jumlah_agunan,
        'nilai_agunan': nilai_agunan,
        'tenor': tenor,
        'usia_agunan': usia_agunan,
        'aset_maya': aset_maya,
        'kenaikan_nilai_property_agunan': kenaikan_nilai_property_agunan,
        'surat_berharga': surat_berharga
    }
    
    if st.button('Dapatkan Prediksi'):
        result = get_prediction(features)
        if result:
            st.write(f"**Prediksi**: {result.get('prediction', 'Tidak Diketahui')}")
            st.write(f"**Skor Prediksi**: {result.get('prediction_score', 'Tidak Diketahui')}")
            st.write(f"**Keterangan**: {result.get('keterangan', 'Tidak Diketahui')}")

if __name__ == '__main__':
    main()
