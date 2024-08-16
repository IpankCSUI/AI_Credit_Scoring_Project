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

def predict_from_file(uploaded_file):
    """Baca file dan kirimkan data untuk prediksi."""
    # Cek ekstensi file
    file_extension = uploaded_file.name.split('.')[-1]

    # Baca file sesuai dengan ekstensi
    if file_extension == 'xlsx':
        df = pd.read_excel(uploaded_file)
    elif file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Format file tidak didukung. Harap unggah file Excel (.xlsx) atau CSV (.csv).")
        return None
    
    predictions = []

    for _, row in df.iterrows():
        features = row.to_dict()
        result = get_prediction(features)
        if result:
            predictions.append(result)

    return predictions

def main():
    st.title("Prediksi Kelayakan Agunan")

    # Navigasi halaman
    page = st.sidebar.selectbox("Pilih Halaman", ["Input Manual", "Unggah File"])

    if page == "Input Manual":
        st.header("Input Manual")
        
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
    
    elif page == "Unggah File":
        st.header("Unggah File")
        uploaded_file = st.file_uploader("Pilih file", type=["xlsx", "csv"])
        
        if uploaded_file:
            predictions = predict_from_file(uploaded_file)
            
            if predictions:
                st.write("**Prediksi dari File**")
                for i, result in enumerate(predictions):
                    st.write(f"**Baris {i+1}**")
                    st.write(f"**Prediksi**: {result.get('prediction', 'Tidak Diketahui')}")
                    st.write(f"**Skor Prediksi**: {result.get('prediction_score', 'Tidak Diketahui')}")
                    st.write(f"**Keterangan**: {result.get('keterangan', 'Tidak Diketahui')}")
            else:
                st.error("Tidak ada prediksi yang dihasilkan.")

if __name__ == '__main__':
    main()
