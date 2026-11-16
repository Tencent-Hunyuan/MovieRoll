// Share one catalog load between the page and its decorative film strips.
(() => {
  let pending;
  window.loadMovieRollContent = () => pending ||= location.protocol === 'file:'
    ? new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'content-data.js';
        script.onload = () => window.MOVIEROLL_CONTENT
          ? resolve(window.MOVIEROLL_CONTENT)
          : reject(new Error('Offline catalog is missing. Rebuild content-data.js.'));
        script.onerror = () => reject(new Error('Cannot load content-data.js. Extract the complete ZIP first.'));
        document.head.append(script);
      })
    : fetch('content.json').then(response => {
        if (!response.ok) throw new Error('Unable to load content.json');
        return response.json();
      });
})();
