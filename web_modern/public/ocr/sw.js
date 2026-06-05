const CACHE_NAME = 'korean-ocr-pwa-v2';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './app.js',
  './styles.css',
  './manifest.json',
  './idx_to_char.json',
  './korean_ocr_quant.onnx'
];

// Install Service Worker - cache essential assets
self.addEventListener('install', (e) => {
  console.log('[Service Worker] Installing version 2...');
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Pre-caching static assets...');
      return cache.addAll(ASSETS_TO_CACHE);
    }).then(() => {
      return self.skipWaiting();
    })
  );
});

// Activate Service Worker - clean up old caches
self.addEventListener('activate', (e) => {
  console.log('[Service Worker] Activating and cleaning old caches...');
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[Service Worker] Removing old cache:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => {
      return self.clients.claim();
    })
  );
});

// Fetch Interceptor - Network First Strategy
// Tries to load from network first. If online, gets the newest files and updates the cache.
// If offline/network fails, falls back to the local cache.
self.addEventListener('fetch', (e) => {
  // Only handle GET requests
  if (e.request.method !== 'GET') return;

  e.respondWith(
    fetch(e.request)
      .then((networkResponse) => {
        // If request was successful, cache it and return
        if (networkResponse && networkResponse.status === 200) {
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(e.request, responseClone);
          });
        }
        return networkResponse;
      })
      .catch(() => {
        // Network failed (offline), fetch from cache
        console.log('[Service Worker] Offline mode: loading from cache for', e.request.url);
        return caches.match(e.request);
      })
  );
});
