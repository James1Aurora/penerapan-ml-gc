const express = require('express');
const multer = require('multer');
const path = require('path');
const { predictHandler, historyHandler, errorHandler } = require('./handler');

// Setup storage untuk multer
const storage = multer.memoryStorage();
const upload = multer({ storage: storage, limits: { fileSize: 1000000 } });

const router = express.Router();

// Route untuk prediksi
router.post('/predict', upload.single('image'), predictHandler);

// Route untuk riwayat prediksi
router.get('/predict/histories', historyHandler);

// Route untuk menangani error file yang terlalu besar
router.use(errorHandler);

module.exports = router;
