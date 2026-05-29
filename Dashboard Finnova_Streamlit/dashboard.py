import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

# ==========================================
# SETUP HALAMAN & KONFIGURASI
# ==========================================
st.set_page_config(page_title="Finova Dashboard", page_icon="💸", layout="wide")
sns.set_theme(style='whitegrid')

st.title("💸 Dashboard Analisis Perilaku Keuangan Pengguna Finova")
st.markdown("Dashboard ini menampilkan hasil analisis EDA perilaku keuangan berdasarkan hasil pengisian kuesioner (*Google Form*).")

# ==========================================
# FUNGSI WRANGLING & FEATURE ENGINEERING
# ==========================================
@st.cache_data
def load_and_clean_data(file_path):
    # Membaca data CSV (Delimiter disesuaikan dengan GForm: titik koma)
    df = pd.read_csv(file_path, sep=';')
    
    # 1. Rename Kolom
    df.columns = [
        'timestamp', 'usia', 'status', 'penghasilan', 'catat_pengeluaran',
        'alat_pencatatan', 'budget', 'persen_habis', 'kategori_pengeluaran',
        'impulsif', 'kehabisan_uang', 'kontrol_pengeluaran', 'penyebab_boros',
        'menabung', 'dana_darurat', 'paylater', 'kesadaran_keuangan',
        'minat_prediksi_ai', 'peringatan_budget', 'simpan_struk',
        'minat_ai_finance', 'kesulitan_keuangan'
    ]
    
    # 2. Cleaning
    df = df.drop_duplicates().dropna()
    cols_numeric = ['catat_pengeluaran', 'impulsif', 'menabung', 'kesadaran_keuangan']
    for col in cols_numeric:
        df[col] = df[col].astype(int)

    # 3. Mapping Kategori menjadi Skor
    budget_map = {
        'tidak punya': 1,
        'ya, tapi sering tidak konsisten': 2,
        'ya, dan saya selalu mengikutinya': 3
    }
    df['budget_score'] = df['budget'].str.lower().str.strip().map(budget_map)
    df['catat_score'] = df['catat_pengeluaran']
    df['impulsif_score'] = df['impulsif']
    df['menabung_score'] = df['menabung']
    df['aware_score'] = df['kesadaran_keuangan']

    # 4. Feature Engineering: Financial Score & Label
    df['financial_score'] = (
        df['catat_score'] + df['menabung_score'] + 
        df['budget_score'] + df['aware_score'] - df['impulsif_score']
    )

    def labeling(score):
        if score >= 10:
            return 'Healthy'
        elif score >= 6:
            return 'Moderate'
        else:
            return 'At Risk'

    df['financial_label'] = df['financial_score'].apply(labeling)
    
    # Menghitung flag biner Dana Darurat (Punya=1, Tidak=0)
    df['punya_dana_darurat'] = (df['dana_darurat'].str.lower() != 'tidak punya').astype(int)
    
    return df

# ==========================================
# LOAD DATA OTOMATIS (TERINTEGRASI GITHUB)
# ==========================================
try:
    # Coba membaca dataset jika berada di folder yang sama (root)
    df = load_and_clean_data("Finova AI (Jawaban).csv")
except FileNotFoundError:
    try:
        # Coba membaca dataset jika berada di dalam folder 'dashboard'
        df = load_and_clean_data("dashboard/Finova AI (Jawaban).csv")
    except FileNotFoundError:
        st.error("Error: File dataset 'Finova AI (Jawaban).csv' tidak ditemukan. Pastikan Anda sudah mengunggahnya ke GitHub dengan nama yang sama persis.")
        st.stop()


# ==========================================
# SIDEBAR FILTER (INTERAKTIF)
# ==========================================
with st.sidebar:
    st.header("🔍 Filter Data")
    
    # Filter Status
    status_list = df['status'].unique().tolist()
    selected_status = st.multiselect("Pilih Status Pekerjaan:", status_list, default=status_list)
    
    # Filter Usia
    usia_list = df['usia'].unique().tolist()
    selected_usia = st.multiselect("Pilih Rentang Usia:", usia_list, default=usia_list)
    
    # Filter Penghasilan
    income_list = df['penghasilan'].unique().tolist()
    selected_income = st.multiselect("Pilih Penghasilan Bulanan:", income_list, default=income_list)

# Menerapkan filter pada dataset
main_df = df[
    (df['status'].isin(selected_status)) & 
    (df['usia'].isin(selected_usia)) & 
    (df['penghasilan'].isin(selected_income))
]

# Jika data kosong setelah difilter, hentikan eksekusi dan beri peringatan
if main_df.empty:
    st.warning("Data tidak ditemukan untuk kombinasi filter yang dipilih. Silakan sesuaikan filter di sidebar.")
    st.stop()

# ==========================================
# PERSIAPAN DATA AGREGAT UNTUK VISUALISASI
# ==========================================
cols_focus = ['catat_score', 'budget_score', 'impulsif_score', 'menabung_score']
label_order = ['Healthy', 'Moderate', 'At Risk']

# MENGGUNAKAN main_df (YANG SUDAH DIFILTER) AGAR INTERAKTIF
agg_df = main_df.groupby('financial_label')[cols_focus].mean().reindex(label_order).reset_index()
agg_dana = main_df.groupby('financial_label')['punya_dana_darurat'].mean().reindex(label_order).reset_index()
agg_dana['punya_dana_darurat'] = agg_dana['punya_dana_darurat'] * 100 

st.markdown("---")

# 1. METRICS (KPI)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Responden Terpilih", value=len(main_df))
with col2:
    st.metric("✅ Status 'Healthy'", value=len(main_df[main_df['financial_label'] == 'Healthy']))
with col3:
    st.metric("⚠️ Status 'Moderate'", value=len(main_df[main_df['financial_label'] == 'Moderate']))
with col4:
    st.metric("🚨 Status 'At Risk'", value=len(main_df[main_df['financial_label'] == 'At Risk']))

st.markdown("---")

# 2. VISUALISASI (Asli dari Kodingan Anda)
st.subheader("📊 Analisis Perilaku Keuangan Berdasarkan Status Finansial")

# Fungsi warna highlight (Biru Muda untuk skor tertinggi, sisanya Abu-abu)
def get_colors(values):
    return ['#72BCD4' if val == max(values) else '#D3D3D3' for val in values]

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Subplot 1: Pencatatan Pengeluaran
sns.barplot(ax=axes[0, 0], x='financial_label', y='catat_score', data=agg_df, palette=get_colors(agg_df['catat_score']))
axes[0, 0].set_title('Rata-rata Skor Pencatatan Pengeluaran', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Skor Rata-rata')
axes[0, 0].set_xlabel('')

# Subplot 2: Pengelolaan Budget
sns.barplot(ax=axes[0, 1], x='financial_label', y='budget_score', data=agg_df, palette=get_colors(agg_df['budget_score']))
axes[0, 1].set_title('Rata-rata Skor Pengelolaan Budget', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Skor Rata-rata')
axes[0, 1].set_xlabel('')

# Subplot 3: Impulsive Spending
sns.barplot(ax=axes[1, 0], x='financial_label', y='impulsif_score', data=agg_df, palette=get_colors(agg_df['impulsif_score']))
axes[1, 0].set_title('Rata-rata Skor Pembelian Impulsif', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Skor Rata-rata')
axes[1, 0].set_xlabel('')

# Subplot 4: Kebiasaan Menabung
sns.barplot(ax=axes[1, 1], x='financial_label', y='menabung_score', data=agg_df, palette=get_colors(agg_df['menabung_score']))
axes[1, 1].set_title('Rata-rata Skor Kebiasaan Menabung', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Skor Rata-rata')
axes[1, 1].set_xlabel('')

# Subplot 5: Kepemilikan Dana Darurat
sns.barplot(ax=axes[2, 0], x='financial_label', y='punya_dana_darurat', data=agg_dana, palette=get_colors(agg_dana['punya_dana_darurat']))
axes[2, 0].set_title('Persentase Kepemilikan Dana Darurat (%)', fontsize=14, fontweight='bold')
axes[2, 0].set_ylabel('Persentase (%)')
axes[2, 0].set_xlabel('')

# Hapus subplot ke-6 yang kosong
fig.delaxes(axes[2, 1])

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ==========================================
# VISUALISASI TAMBAHAN BARU
# ==========================================
st.subheader("📈 Analisis Lanjutan: Proporsi, Korelasi, dan Sentimen")

# ROW A: Pie Chart & Heatmap
col_pie, col_heat = st.columns(2)

with col_pie:
    st.markdown("**Proporsi Status Kesehatan Finansial**")
    label_counts = main_df['financial_label'].value_counts()
    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
    colors_pie = ['#72BCD4', '#AAB7B8', '#D3D3D3'] 
    ax_pie.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=colors_pie)
    ax_pie.axis('equal') 
    st.pyplot(fig_pie)

with col_heat:
    st.markdown("**Heatmap Korelasi Perilaku Keuangan**")
    corr_cols = ['catat_score', 'budget_score', 'impulsif_score', 'menabung_score', 'aware_score']
    corr_matrix = main_df[corr_cols].corr()
    fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", ax=ax_heat)
    st.pyplot(fig_heat)

# ROW B: Bar Chart Sentimen & Word Cloud
col_sent, col_word = st.columns(2)

with col_sent:
    st.markdown("**Analisis Sentimen: Hambatan Utama Pengguna**")
    # Mapping sederhana berdasarkan kata kunci dari kolom kesulitan_keuangan
    sentimen_map = {'Sulit / Susah': 0, 'Goda Diskon / Promo': 0, 'Lainnya': 0}
    for text in main_df['kesulitan_keuangan'].astype(str):
        if 'sulit' in text.lower() or 'susah' in text.lower():
            sentimen_map['Sulit / Susah'] += 1
        elif 'diskon' in text.lower() or 'promo' in text.lower():
            sentimen_map['Goda Diskon / Promo'] += 1
        else:
            sentimen_map['Lainnya'] += 1
            
    sent_df = pd.DataFrame(list(sentimen_map.items()), columns=['Kategori', 'Jumlah'])
    fig_sent, ax_sent = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Kategori', y='Jumlah', data=sent_df, palette=['#72BCD4', '#AAB7B8', '#D3D3D3'], ax=ax_sent)
    ax_sent.set_ylabel("Jumlah Keluhan")
    ax_sent.set_xlabel("")
    st.pyplot(fig_sent)

with col_word:
    st.markdown("**Word Cloud: Keluhan Kesulitan Keuangan**")
    text_data = " ".join(main_df['kesulitan_keuangan'].astype(str)).lower()
    
    if text_data.strip() and len(text_data) > 5:
        # 1. Mendefinisikan Stop Words Bahasa Indonesia + Kata Konteks yang tidak perlu
        id_stopwords = set([
            "dan", "yang", "di", "ke", "dari", "untuk", "pada", "dalam", 
            "dengan", "itu", "ini", "karena", "jika", "atau", "tidak", 
            "ada", "juga", "kalau", "buat", "saat", "lebih", "saya", 
            "aku", "sih", "nya", "ya", "aja", "saja", "bisa", "jadi", 
            "kadang", "kan", "terus", "hal", "sama", "sudah", "belum",
            "uang", "keuangan", "pengeluaran", "mengatur", "mengelola" # Kata konteks
        ])
        
        # 2. Memasukkan id_stopwords ke parameter stopwords
        wordcloud = WordCloud(
            width=800, 
            height=500, 
            background_color='white', 
            colormap='Blues',
            stopwords=id_stopwords 
        ).generate(text_data)
        
        fig_word, ax_word = plt.subplots(figsize=(8, 5))
        ax_word.imshow(wordcloud, interpolation='bilinear')
        ax_word.axis("off")
        st.pyplot(fig_word)
    else:
        st.info("Data teks tidak mencukupi untuk membuat Word Cloud dari filter saat ini.")

st.markdown("---")

# 3. INSIGHT & REKOMENDASI (Asli dari Kodingan Anda)
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("💡 Insight")
    st.info("""
    1. **Disiplin Pengeluaran & Budget:** Batang biru tertinggi pada grafik pencatatan dan *budget* didominasi oleh pengguna berkategori **Healthy**. Hal ini mengonfirmasi bahwa pengelolaan arus kas yang ketat berkorelasi langsung dengan kesehatan finansial yang baik.
    2. **Impulsive Spending:** Sangat terlihat jelas bahwa pengguna dalam kategori **At Risk** memiliki kecenderungan melakukan pembelian impulsif atau tergoda diskon paling tinggi dibandingkan kelompok lainnya.
    3. **Menabung & Dana Darurat:** Pengguna **Healthy** dan **Moderate** secara signifikan mendominasi kebiasaan menabung. Persentase dana darurat mencapai puncaknya pada kelompok *Healthy*, sementara kelompok *At Risk* hampir tidak memilikinya.
    """)

with col_b:
    st.subheader("🎯 Rekomendasi Finova")
    st.success("""
    1. **Smart Budget Guardian (Anti-Impulsif):** Karena kelompok *At Risk* sangat rentan terhadap *impulsive spending*, Finova perlu mengembangkan fitur notifikasi proaktif (*early warning*) yang memblokir mental impulsif saat mendekati batas *budget*.
    2. **Gamifikasi Pencatatan:** Untuk mengubah perilaku kelompok *At Risk*, aplikasi dapat menerapkan sistem *reward* (poin/lencana) bagi pengguna yang konsisten mencatat pengeluaran (*streak*).
    3. **Alokasi Dana Darurat AI:** AI Finova dapat menganalisis sisa uang di akhir bulan dan memberikan rekomendasi *pop-up* untuk langsung mengalihkan sisa tersebut ke pos Dana Darurat.
    """)

st.caption("Hak Cipta © 2026 - Analisis Proyek Finova AI")
