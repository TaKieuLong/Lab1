const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Cấu hình token API và URL của Hugging Face Translation API
const HF_API_TOKEN = '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'; // Thay bằng token của bạn
const HF_API_URL = 'https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-vi';

app.use(bodyParser.json());
app.use(cors());
// Phục vụ các file tĩnh từ thư mục public
app.use(express.static(path.join(__dirname, 'public')));

// Endpoint dịch thuật: nhận văn bản tiếng Anh và trả về bản dịch tiếng Việt
app.post('/translate', async (req, res) => {
    const { text } = req.body;
    if (!text) {
        return res.status(400).json({ error: 'No text provided' });
    }

    try {
        const response = await axios.post(
            HF_API_URL,
            { inputs: text },
            {
                headers: {
                    'Authorization': `Bearer ${HF_API_TOKEN}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        // Giả sử API trả về mảng chứa đối tượng có key "translation_text"
        // Ví dụ: [ { translation_text: "Bản dịch tiếng Việt" } ]
        const translationData = response.data;
        if (Array.isArray(translationData) && translationData.length > 0 && translationData[0].translation_text) {
            res.json({ translation: translationData[0].translation_text });
        } else {
            res.status(500).json({ error: 'Unexpected response format', details: translationData });
        }
    } catch (error) {
        res.status(error.response ? error.response.status : 500).json({
            error: 'Error calling Hugging Face API',
            details: error.response ? error.response.data : error.message
        });
    }
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
