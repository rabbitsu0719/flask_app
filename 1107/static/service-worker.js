const cacheName = 'my-cache-v1';
const contentToCache = [
 '/'
];


// 설치 시 캐시 저장
self.addEventListener('install', event => {
 event.waitUntil(
   caches.open(cacheName).then(cache => {
     return cache.addAll(contentToCache);
   })
 );
});


// 활성화 시 구버전 캐시 삭제
self.addEventListener('activate', event => {
 event.waitUntil(
   caches.keys().then(cacheNames => {
     return Promise.all(
       cacheNames.map(name => {
         if (name !== cacheName) {
           return caches.delete(name);
         }
       })
     );
   })
 );
});


// 네트워크 요청 가로채기
self.addEventListener('fetch', event => {
 event.respondWith(
   caches.match(event.request).then(response => {
     return response || fetch(event.request).then(fetchResponse => {
       return caches.open(cacheName).then(cache => {
         cache.put(event.request, fetchResponse.clone());
         return fetchResponse;
       });
     });
   })
 );
});
