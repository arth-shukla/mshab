window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function () {
	// Check for click events on the navbar burger icon

	var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
	}

	// Initialize all div with carousel class
	var carousels = bulmaCarousel.attach('.carousel', options);

	bulmaSlider.attach();
	document.getElementById('copyButton').addEventListener('click', copyText);

	// Add scroll handler for tables
	const tables = document.querySelectorAll('table');
	tables.forEach(table => {
		updateTableShadows(table);
		table.addEventListener('scroll', () => updateTableShadows(table));
	});
})

function updateTableShadows(table) {
	const maxScroll = table.scrollWidth - table.clientWidth;
	const scrollLeft = table.scrollLeft;
	
	let background = [];
	
	// Add left shadow if not at start
	if (scrollLeft > 0) {
		background.push('radial-gradient(50% 50% at 0 50%, rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0))');
	}
	
	// Add right shadow if not at end
	if (scrollLeft < maxScroll) {
		background.push('radial-gradient(50% 50% at 100% 50%, rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0)) 100% 0');
	}
	
	// Always maintain the white gradients for smooth transitions
	background.unshift(
		'linear-gradient(to right, white 30%, rgba(255, 255, 255, 0))',
		'linear-gradient(to right, rgba(255, 255, 255, 0), white 70%) 100% 0'
	);
	
	table.style.background = background.join(',');
	table.style.backgroundRepeat = 'no-repeat';
	table.style.backgroundSize = '60px 100%';
	table.style.backgroundAttachment = 'local, local, scroll, scroll';
}

const copyText = () => {
	const text = `@inproceedings{shukla2025maniskillhab,
	author       = {Arth Shukla and Stone Tao and Hao Su},
	title        = {ManiSkill-HAB: {A} Benchmark for Low-Level Manipulation in Home Rearrangement Tasks},
	booktitle    = {The Thirteenth International Conference on Learning Representations, {ICLR} 2025, Singapore, April 24-28, 2025},
	publisher    = {OpenReview.net},
	year         = {2025},
	url          = {https://openreview.net/forum?id=6bKEWevgSd},
	timestamp    = {Thu, 15 May 2025 17:19:05 +0200},
	biburl       = {https://dblp.org/rec/conf/iclr/ShuklaTS25.bib},
	bibsource    = {dblp computer science bibliography, https://dblp.org}
}`;

	navigator.clipboard.writeText(text).then(() => {
		const messageDiv = document.getElementById('message');
		messageDiv.textContent = 'Copied to clipboard';
		messageDiv.style.color = '#5a5aFa';
		messageDiv.style.paddingRight = '10px';

		setTimeout(() => {
			messageDiv.textContent = '';
		}, 2000);
	}).catch(err => {
		console.error('Failed to copy: ', err);
	});
};

