(() => {
  const root = document.documentElement;
  const progress = document.querySelector('.reading-progress span');
  const sidebar = document.querySelector('.sidebar');
  const navToggle = document.querySelector('.nav-toggle');
  const themeToggle = document.querySelector('.theme-toggle');
  const search = document.querySelector('#toc-search');
  const tocLinks = [...document.querySelectorAll('#toc a')];
  const headings = [...document.querySelectorAll('article h2[id], article h3[id]')];

  const savedTheme = localStorage.getItem('ws-book-theme');
  if (savedTheme === 'light' || savedTheme === 'dark') root.dataset.theme = savedTheme;

  themeToggle.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('ws-book-theme', root.dataset.theme);
  });

  navToggle.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', String(open));
  });
  tocLinks.forEach((link) => link.addEventListener('click', () => {
    sidebar.classList.remove('open');
    navToggle.setAttribute('aria-expanded', 'false');
  }));

  search.addEventListener('input', () => {
    const query = search.value.trim().toLocaleLowerCase('zh-CN');
    document.querySelectorAll('#toc li').forEach((item) => {
      const ownLink = item.querySelector(':scope > a');
      const childMatch = [...item.querySelectorAll('ul a')]
        .some((link) => link.textContent.toLocaleLowerCase('zh-CN').includes(query));
      const ownMatch = ownLink && ownLink.textContent.toLocaleLowerCase('zh-CN').includes(query);
      item.hidden = Boolean(query) && !ownMatch && !childMatch;
    });
  });

  const update = () => {
    const height = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = `${height > 0 ? Math.min(100, scrollY / height * 100) : 0}%`;
    let active = headings[0];
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top <= 140) active = heading;
      else break;
    }
    tocLinks.forEach((link) => link.classList.toggle('active', active && link.hash === `#${active.id}`));
  };
  addEventListener('scroll', update, {passive: true});
  update();

  document.querySelectorAll('pre').forEach((pre) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-code';
    button.textContent = '复制';
    button.addEventListener('click', async () => {
      await navigator.clipboard.writeText(pre.innerText);
      button.textContent = '已复制';
      setTimeout(() => { button.textContent = '复制'; }, 1200);
    });
    pre.appendChild(button);
  });
})();
