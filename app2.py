import streamlit as st
st.title("🏠 Rental Property Data Analysis - India")
upload_file=st.file_uploader
df=pd.read_csv(uploaded_file)
st.success()
st.write("Preview of Dataset:",df.head())
st.checkbox("Show Missing Values")
st.write(df.isnull().sum())
fill_option=st.selectbox("Fill missing values with:",["0","-1","Custom"])
st.checkbox()
st.selectbox()
st.header("🔹 Data Cleaning")
st.header("🔹 Feature Engineering")
st.header("📊 Visualizations")
plt.hist(df['rent_month'], bins=30, color='skyblue', edgecolor='black')
# ... more plotting code ...
st.pyplot(plt.gcf())
st.pyplot() 
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Cleaned Data as CSV",
    data=csv,
    file_name='cleaned_rental_data.csv',
    mime='text/csv',
)
```
st.download_button()` 



