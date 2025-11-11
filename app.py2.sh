import streamlit as st
st.title("🏠 Rental Property Data Analysis - India")
uploaded_file = st.file_uploader
df = pd.read_csv(uploaded_file)("Upload your CSV file", type=["Rental Property Data Analysis - India"])
st.success
st.write
st.write("Preview of dataset:", df.head())
st.checkbox
st.write(df.isnull().sum())
fill_option = st.selectbox
st.checkbox()
st.selectbox()
st.header
plt.figure(figsize=(10, 6))
plt.hist(df['rent_month'], bins=30, color='skyblue', edgecolor='black')
# ... more plotting code 
st.pyplot(plt.gcf())
st.pyplot()
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Cleaned Data as CSV",
    data=csv,
    file_name='cleaned_rental_data.csv',
    mime='text/csv',
)
st.download_button()


