# 🎬 Movie Genre Predictor

An AI-powered Streamlit application that predicts movie genres based on plot summaries using deep learning and natural language processing.

## ✨ Features

- **🎯 Genre Prediction**: Predicts multiple genres with confidence scores
- **🌐 Multi-language Support**: Translate and convert summaries to audio in 6 languages
- **📊 Model Evaluation**: View detailed performance metrics and confusion matrices
- **🎵 Audio Generation**: Convert plot summaries to speech with translation
- **📈 Interactive Visualizations**: Beautiful charts and graphs
- **📋 Prediction History**: Track and review previous predictions

## 🚀 Supported Languages

| Language | Translation | Audio | Status |
|----------|-------------|-------|---------|
| **English** 🇺🇸 | N/A | ✅ | Full support |
| **Arabic** 🇦🇪 | ✅ DeepL | ✅ gTTS | Full support |
| **French** 🇫🇷 | ✅ DeepL | ✅ gTTS | Full support |
| **Spanish** 🇪🇸 | ✅ DeepL | ✅ gTTS | Full support |
| **Korean** 🇰🇷 | ✅ DeepL | ✅ gTTS | Full support |
| **Urdu** 🇵🇰 | ❌ DeepL | ✅ gTTS | Audio only |

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-genre-predictor.git
   cd movie-genre-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit torch pandas plotly gtts nltk pillow aiohttp
   ```

3. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. **Run the application**
   ```bash
   streamlit run app_new.py
   ```

## 📁 Project Structure

```
movie-genre-predictor/
├── app_new.py                    # Main Streamlit application
├── audio_converter.py            # Audio conversion utility
├── model.pt                      # Neural network model (20MB)
├── tfidf.pkl                     # TF-IDF vectorizer
├── mlb.pkl                       # Multi-label binarizer
├── genre.pkl                     # Genre model components
├── genre_mlb.pkl
├── genre_models.pkl
├── genre_vectorizer.pkl
├── confusion_matrix/             # Evaluation visualizations
├── audio_output/                 # Generated audio files
├── audio_summaries/              # Audio summaries
├── temp_backup/                  # Temporary backup files
├── cleaned_movie_metadata.csv    # Movie metadata
├── cleaned_plot_summaries.csv    # Plot summaries
├── plot_summaries.txt            # Raw plot data
└── movie.metadata.tsv            # Original metadata
```

## 🎯 Model Performance

- **F1 Score (Micro)**: 60.77%
- **Precision (Micro)**: 54.00%
- **Recall (Micro)**: 69.48%
- **Supported Genres**: 18 different movie genres
- **Architecture**: Neural network with TF-IDF features

## 🔧 Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: PyTorch, scikit-learn
- **NLP**: NLTK, TF-IDF
- **Translation**: DeepL API
- **Text-to-Speech**: Google TTS (gTTS)
- **Visualization**: Plotly
- **Data Processing**: Pandas

## 🌟 Key Features

### Genre Prediction
- Input movie plot summaries
- Get predicted genres with confidence scores
- View interactive visualizations
- Track prediction history

### Audio Conversion
- Translate summaries to multiple languages
- Generate audio with different accents
- Download audio files
- Adjust speech speed and volume

### Model Analysis
- View detailed performance metrics
- Explore confusion matrices
- Analyze genre-specific performance
- Interactive charts and graphs

## 📊 API Integration

- **DeepL API**: For text translation
- **Google TTS**: For audio generation
- **Custom Neural Network**: For genre prediction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DeepL for translation services
- Google TTS for audio generation
- Streamlit for the web framework
- PyTorch for deep learning capabilities

## 📞 Support

If you have any questions or issues, please open an issue on GitHub.

---

**Made with ❤️ using AI and Machine Learning** 