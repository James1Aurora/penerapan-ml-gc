const express = require('express');
const cors = require('cors');
const { loadModel } = require('./handler');
const routes = require('./routes');

// Membuat Express server
const app = express();

// Middleware untuk CORS dan body parsing
app.use(cors());
app.use(express.json());

// Menambahkan routes
app.use('/api', routes);

// Menjalankan server dan memuat model
const PORT = process.env.PORT || 8080;
app.listen(PORT, async () => {
  console.log(`Server running on port ${PORT}`);
  await loadModel(); // Memuat model setelah server berjalan
});
