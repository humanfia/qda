// The QDA docs have one white scheme: pin furo's theme to light, whatever the browser prefers.
localStorage.setItem("theme", "light");
const pinLight = () => { document.body.dataset.theme = "light"; };
if (document.body) pinLight(); else document.addEventListener("DOMContentLoaded", pinLight);
