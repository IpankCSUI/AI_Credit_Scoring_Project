import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

# Load your original dataset
df_baru_original = pd.read_csv('path_to_your_dataset/df_baru_original.csv')

# Kolom-kolom untuk ordinal encoding
ordinal_columns = ['tipe_agunan', 'kondisi_agunan', 'kemudahan_dijual_kembali', 'status_lunas_agunan',
                   'agunan_dibanyak_pinjaman', 'kesediaan_untuk_digadaikan', 'jenis_asset_bergerak_atau_tidak',
                   'riwayat_agunan_sebelumnya', 'status_kepemilikan_asset', 'asuransi_agunan']

# Kolom-kolom untuk label encoding
nominal_columns = ['pekerjaan', 'tujuan_pinjaman', 'agunan', 'merk_aset']

# Mapping untuk ordinal encoding
categories = {
    'tipe_agunan': ['tak berwujud', 'berwujud'],
    'kondisi_agunan': ['Resiko Tinggi', 'Sangat Lemah', 'Lemah', 'Meragukan', 'Cukup', 'Kuat', 'Sangat Kuat'],
    'kemudahan_dijual_kembali': ['resiko tinggi', 'sangat lemah', 'lemah', 'meragukan', 'cukup', 'kuat', 'sangat kuat'],
    'status_lunas_agunan': ['Belum Lunas', 'Lunas'],
    'agunan_dibanyak_pinjaman': ['Tidak', 'Ya'],
    'kesediaan_untuk_digadaikan': ['Tidak', 'Ya'],
    'jenis_asset_bergerak_atau_tidak': ['Tidak', 'Ya'],
    'riwayat_agunan_sebelumnya': ['Tidak Pernah', 'Pernah'],
    'status_kepemilikan_asset': ['Non-Pribadi', 'Pribadi'],
    'asuransi_agunan': ['False', 'True'],
}

# Buat ordinal encoder dengan mapping categories dan menangani unknown values
ordinal_encoder = OrdinalEncoder(
    categories=[categories[col] for col in ordinal_columns],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

# Custom transformer untuk Label Encoding kolom nominal
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.encoders = {col: LabelEncoder().fit(X[col]) for col in self.columns}
        return self
    
    def transform(self, X):
        output = X.copy()
        for col in self.columns:
            output[col] = self.encoders[col].transform(X[col])
        return output

label_encoder = MultiColumnLabelEncoder(columns=nominal_columns)

# Column transformer untuk menggabungkan kedua jenis encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', ordinal_encoder, ordinal_columns),
        ('nom', label_encoder, nominal_columns)
    ],
    remainder='passthrough'  # Menambahkan kolom lainnya ke hasil akhir
)

# Latih preprocessor pada dataset
preprocessor.fit(df_baru_original)

# Simpan preprocessor ke file
preprocessor_path = 'path_to_save_preprocessor/preprocessor.pkl'
with open(preprocessor_path, 'wb') as f:
    pickle.dump(preprocessor, f)
