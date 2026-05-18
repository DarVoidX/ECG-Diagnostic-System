/**
 * CARDIO.AI — Neural ECG Classification System
 * Fixed: markers on chart, upload flow, trimmed padding, image report with all leads
 */

const API_BASE = 'http://localhost:5000/api';
const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

const MARKER_COLORS = {
    pWave: '#10b981', rPeak: '#f59e0b', tWave: '#3b82f6', qrs: '#ef4444', trace: '#000000'
};

const CLASS_SHORT = {
    'Normal ECG': 'NORMAL', 'Myocardial Infarction (MI)': 'MI',
    'Conduction Disturbance (CD)': 'CD', 'ST/T Change (STTC)': 'STTC', 'Hypertrophy (HYP)': 'HYP',
};

let currentResults = null, currentLead = 'II', currentRecordId = '', ecgChart = null, heaFile = null, datFile = null;
let isDark = false;

// ── DISEASE DETAIL DATA ──
const DISEASE_DATA = {
    'Myocardial Infarction (MI)': {
        shortName: 'MI',
        fullName: 'MYOCARDIAL\nINFARCTION',
        image: 'images/mi.png',
        subtitle: 'Myocardial Infarction (MI) occurs when blood flow to a part of the heart muscle is blocked for a prolonged period, causing tissue damage or death. It is a life-threatening cardiac emergency requiring immediate intervention.',
        overview: 'A myocardial infarction, commonly known as a heart attack, results from the sudden occlusion of a coronary artery — typically by a thrombus forming over a ruptured atherosclerotic plaque. The resulting ischemia leads to necrosis of the myocardium if perfusion is not promptly restored. MI is classified by ECG findings as ST-Elevation MI (STEMI) or Non-ST-Elevation MI (NSTEMI).',
        causes: [
            'Coronary artery atherosclerosis and plaque rupture',
            'Coronary artery spasm (Prinzmetal angina)',
            'Thrombotic occlusion of coronary vessels',
            'Hypertension and chronic high blood pressure',
            'Hyperlipidemia (elevated LDL cholesterol)',
            'Diabetes mellitus and insulin resistance',
            'Smoking and tobacco use',
            'Family history of premature cardiovascular disease',
            'Obesity and sedentary lifestyle',
            'Cocaine or amphetamine abuse'
        ],
        symptoms: [
            'Severe, crushing chest pain (retrosternal) lasting >20 minutes',
            'Radiation of pain to left arm, jaw, neck, or back',
            'Diaphoresis (profuse sweating)',
            'Dyspnea (shortness of breath)',
            'Nausea, vomiting, and lightheadedness',
            'Anxiety and sense of impending doom',
            'Pallor and cold, clammy skin',
            'Syncope or near-syncope in severe cases',
            'Silent MI possible in diabetic or elderly patients'
        ],
        ecgSigns: [
            'ST-segment elevation ≥1mm in limb leads or ≥2mm in precordial leads',
            'Pathological Q-waves indicating transmural necrosis',
            'T-wave inversion in affected leads',
            'Reciprocal ST-depression in contralateral leads',
            'New left bundle branch block (LBBB)',
            'Loss of R-wave progression in precordial leads'
        ],
        treatment: [
            'Immediate dual antiplatelet therapy (Aspirin + P2Y12 inhibitor)',
            'Primary percutaneous coronary intervention (PCI) within 90 minutes',
            'Fibrinolytic therapy if PCI unavailable within 120 minutes',
            'Anticoagulation with heparin (UFH or enoxaparin)',
            'Beta-blockers to reduce myocardial oxygen demand',
            'ACE inhibitors for ventricular remodeling prevention',
            'Statin therapy for plaque stabilization',
            'Cardiac rehabilitation program post-discharge',
            'Coronary artery bypass grafting (CABG) for multi-vessel disease'
        ],
        clinicalNote: 'Time is critical in MI management. The "door-to-balloon" time target is <90 minutes. Every 30-minute delay in reperfusion increases mortality by approximately 7.5%. This AI classification serves as a screening tool — clinical correlation with troponin levels and patient symptoms is mandatory.'
    },
    'Conduction Disturbance (CD)': {
        shortName: 'CD',
        fullName: 'CONDUCTION\nDISORDER',
        image: 'images/cd.png',
        subtitle: 'Conduction Disorders (CD) involve disruption of the normal electrical impulse propagation through the heart\'s conduction system, leading to delayed or blocked signal transmission between cardiac chambers.',
        overview: 'Cardiac conduction disorders encompass a range of abnormalities where the electrical impulse that coordinates heartbeats is delayed or blocked at various points in the conduction system. This includes conditions like bundle branch blocks (RBBB, LBBB), atrioventricular blocks (first, second, and third degree), fascicular blocks, and intraventricular conduction delays. These disorders can range from benign findings to life-threatening arrhythmias requiring pacemaker implantation.',
        causes: [
            'Degenerative fibrosis of the conduction system (aging)',
            'Ischemic heart disease and prior myocardial infarction',
            'Cardiomyopathy (dilated, hypertrophic, or restrictive)',
            'Congenital heart defects and inherited channelopathies',
            'Infectious diseases (Lyme disease, Chagas disease, endocarditis)',
            'Medications (beta-blockers, calcium channel blockers, digoxin)',
            'Electrolyte imbalances (hyperkalemia, hypokalemia)',
            'Post-cardiac surgery complications',
            'Infiltrative diseases (amyloidosis, sarcoidosis)',
            'Autoimmune conditions (SLE, rheumatoid arthritis)'
        ],
        symptoms: [
            'Bradycardia (abnormally slow heart rate)',
            'Syncope or pre-syncope (fainting or near-fainting)',
            'Dizziness and lightheadedness',
            'Exercise intolerance and fatigue',
            'Palpitations or irregular heartbeat sensation',
            'Dyspnea on exertion',
            'Chest discomfort in some cases',
            'Stokes-Adams attacks (sudden LOC with complete heart block)',
            'May be completely asymptomatic in mild cases'
        ],
        ecgSigns: [
            'Prolonged PR interval (>200ms) in first-degree AV block',
            'Progressive PR prolongation with dropped beats (Mobitz Type I)',
            'Intermittent dropped QRS complexes (Mobitz Type II)',
            'Complete AV dissociation in third-degree heart block',
            'Wide QRS complex (>120ms) in bundle branch blocks',
            'RSR\' pattern in V1-V2 (RBBB) or broad notched R in I, aVL, V5-V6 (LBBB)'
        ],
        treatment: [
            'Discontinuation of offending medications',
            'Correction of underlying electrolyte abnormalities',
            'Temporary transcutaneous or transvenous pacing for emergencies',
            'Permanent pacemaker implantation for symptomatic high-grade blocks',
            'Cardiac resynchronization therapy (CRT) for LBBB with heart failure',
            'Treatment of underlying infections (antibiotics for Lyme disease)',
            'Regular Holter monitoring for intermittent conduction disturbances',
            'Implantable cardioverter-defibrillator (ICD) if associated with VT risk'
        ],
        clinicalNote: 'Conduction disorders range from benign incidental findings (first-degree AV block) to life-threatening emergencies (complete heart block). Clinical significance depends on the level and degree of block, presence of symptoms, and underlying etiology. Always correlate with hemodynamic status.'
    },
    'ST/T Change (STTC)': {
        shortName: 'STTC',
        fullName: 'ST-T WAVE\nCHANGES',
        image: 'images/sttc.png',
        subtitle: 'ST-T Changes (STTC) represent alterations in the ST segment and T wave morphology on ECG, indicating abnormalities in ventricular repolarization that may signify ischemia, electrolyte disorders, or other cardiac pathology.',
        overview: 'ST-T wave abnormalities are among the most common and clinically significant ECG findings. The ST segment represents early ventricular repolarization, while the T wave represents completion of repolarization. Changes in these waveforms can indicate myocardial ischemia, pericarditis, electrolyte disturbances, drug effects, or ventricular strain. Interpretation requires careful clinical correlation as both benign variants and life-threatening conditions can produce similar patterns.',
        causes: [
            'Myocardial ischemia and acute coronary syndrome',
            'Pericarditis (diffuse ST elevation with PR depression)',
            'Left ventricular hypertrophy with strain pattern',
            'Electrolyte imbalances (hypokalemia, hypocalcemia, hyperkalemia)',
            'Digitalis effect and drug-induced changes',
            'Takotsubo (stress) cardiomyopathy',
            'Pulmonary embolism with right heart strain',
            'Hypothermia (Osborn/J waves)',
            'Early repolarization (benign variant)',
            'Myocarditis and cardiomyopathies'
        ],
        symptoms: [
            'May be asymptomatic (incidental finding)',
            'Anginal chest pain (ischemia-related ST changes)',
            'Sharp, pleuritic chest pain (pericarditis)',
            'Dyspnea and exercise intolerance',
            'Palpitations and irregular heartbeat',
            'Fatigue and generalized weakness',
            'Syncope in severe ischemia',
            'Symptoms vary based on underlying etiology'
        ],
        ecgSigns: [
            'ST-segment depression (horizontal or downsloping) suggesting ischemia',
            'ST-segment elevation (convex or concave) in acute injury',
            'T-wave inversion (symmetric deep inversions suggest ischemia)',
            'T-wave flattening in early ischemia or electrolyte disturbance',
            'Diffuse ST elevation with PR depression in pericarditis',
            'Strain pattern: asymmetric ST depression and T inversion in LVH'
        ],
        treatment: [
            'Urgent coronary angiography if acute ischemia is suspected',
            'Antianginal therapy (nitrates, beta-blockers, CCBs)',
            'Correction of underlying electrolyte abnormalities',
            'NSAIDs and colchicine for pericarditis',
            'Anticoagulation for pulmonary embolism',
            'Serial ECG monitoring and troponin levels',
            'Stress testing for risk stratification in stable patients',
            'Revascularization (PCI or CABG) for significant coronary disease'
        ],
        clinicalNote: 'ST-T changes are non-specific and must be interpreted in the clinical context. Comparison with prior ECGs is essential. A new ST change in the setting of chest pain should be treated as acute coronary syndrome until proven otherwise. Early repolarization is a diagnosis of exclusion.'
    },
    'Hypertrophy (HYP)': {
        shortName: 'HYP',
        fullName: 'CARDIAC\nHYPERTROPHY',
        image: 'images/hyp.png',
        subtitle: 'Cardiac Hypertrophy (HYP) involves thickening of the heart muscle walls, typically in response to chronic pressure or volume overload. It increases the risk of arrhythmias, heart failure, and sudden cardiac death.',
        overview: 'Cardiac hypertrophy refers to the abnormal thickening of the heart\'s muscular walls, most commonly affecting the left ventricle (LVH) but also potentially involving the right ventricle (RVH) or atria. LVH develops as a compensatory response to sustained pressure overload (hypertension, aortic stenosis) or volume overload (aortic/mitral regurgitation). While initially adaptive, prolonged hypertrophy leads to diastolic dysfunction, arrhythmias, and eventual systolic failure.',
        causes: [
            'Systemic hypertension (most common cause of LVH)',
            'Aortic valve stenosis or insufficiency',
            'Hypertrophic cardiomyopathy (HCM, genetic)',
            'Coarctation of the aorta',
            'Chronic kidney disease',
            'Pulmonary hypertension (cause of RVH)',
            'Chronic lung disease (COPD, for RVH)',
            'Athletic heart syndrome (physiological hypertrophy)',
            'Obesity and metabolic syndrome',
            'Sleep apnea (contributes to LVH and RVH)'
        ],
        symptoms: [
            'Often asymptomatic in early stages',
            'Exertional dyspnea (shortness of breath)',
            'Chest pain or angina-like symptoms',
            'Palpitations and arrhythmias',
            'Syncope, especially during exertion (HCM)',
            'Orthopnea and paroxysmal nocturnal dyspnea',
            'Lower extremity edema in right heart failure',
            'Exercise intolerance and progressive fatigue',
            'Sudden cardiac death (risk in severe HCM)'
        ],
        ecgSigns: [
            'Increased QRS voltage (Sokolow-Lyon: SV1 + RV5 ≥35mm for LVH)',
            'Left axis deviation in LVH',
            'ST-T strain pattern (asymmetric ST depression + T inversion in lateral leads)',
            'Left atrial enlargement (wide, notched P waves in lead II)',
            'Right axis deviation in RVH',
            'Tall R wave in V1 with right axis deviation (RVH)',
            'Deep S waves in V5-V6 in RVH'
        ],
        treatment: [
            'Aggressive blood pressure control (target <130/80 mmHg)',
            'ACE inhibitors or ARBs (first-line for LVH regression)',
            'Beta-blockers for rate control and symptom relief',
            'Diuretics for volume overload management',
            'Surgical myectomy for obstructive HCM',
            'Alcohol septal ablation as alternative to surgery for HCM',
            'ICD implantation for high-risk sudden cardiac death patients',
            'Aortic valve replacement for severe aortic stenosis',
            'Lifestyle: sodium restriction, weight management, regular exercise'
        ],
        clinicalNote: 'LVH is an independent risk factor for cardiovascular morbidity and mortality. ECG has moderate sensitivity (~50%) but high specificity for LVH. Echocardiography remains the gold standard for quantifying ventricular mass. Regression of LVH with antihypertensive therapy is associated with improved outcomes.'
    },
    'Normal ECG': {
        shortName: 'NORM',
        fullName: 'NORMAL\nSINUS RHYTHM',
        image: 'images/norm.png',
        subtitle: 'A normal ECG indicates that the heart\'s electrical activity is following the expected pattern — regular sinus rhythm with normal intervals, axis, and morphology across all 12 leads.',
        overview: 'A normal electrocardiogram demonstrates organized electrical activity originating from the sinoatrial (SA) node, propagating through the atria (P wave), traversing the atrioventricular (AV) node with appropriate delay (PR interval), and depolarizing the ventricles via the His-Purkinje system (QRS complex) followed by orderly repolarization (T wave). All intervals fall within normal limits and no pathological changes are detected.',
        causes: [
            'Healthy cardiac conduction system',
            'Normal sinus node automaticity (60-100 BPM)',
            'Intact AV nodal conduction',
            'Normal ventricular myocardium',
            'Balanced electrolyte levels',
            'Adequate coronary perfusion',
            'Normal cardiac structure and valve function'
        ],
        symptoms: [
            'No cardiac symptoms expected',
            'Normal exercise tolerance',
            'Regular and comfortable heartbeat',
            'No chest pain, dyspnea, or syncope',
            'Stable hemodynamic status',
            'Note: A normal ECG does not exclude all cardiac pathology'
        ],
        ecgSigns: [
            'Regular R-R intervals with rate 60-100 BPM',
            'Upright P waves in leads I, II, aVF (sinus origin)',
            'PR interval 120-200ms',
            'QRS duration <120ms',
            'Normal QRS axis (-30° to +90°)',
            'No ST elevation or depression',
            'Upright T waves in most leads',
            'QTc interval 350-440ms'
        ],
        treatment: [
            'No specific cardiac treatment required',
            'Maintain cardiovascular fitness with regular exercise',
            'Heart-healthy diet (Mediterranean, DASH)',
            'Regular health check-ups and preventive screening',
            'Blood pressure and cholesterol monitoring',
            'Smoking cessation if applicable',
            'Stress management and adequate sleep',
            'Annual ECG screening for high-risk populations'
        ],
        clinicalNote: 'A normal ECG is reassuring but does not completely exclude cardiac disease. Conditions such as paroxysmal arrhythmias, early-stage cardiomyopathy, or structurally normal channelopathies may present with normal resting ECG. Clinical judgment based on symptoms, risk factors, and additional testing remains essential.'
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initNavigation(); initThemeToggle(); initUpload(); initLiveAIFeed(); initFooterEcgAnimation();
    document.getElementById('download-report-btn')?.addEventListener('click', downloadReport);
});

// ── FOOTER ECG ANIMATION — Realistic moving green signal ──
function initFooterEcgAnimation() {
    const canvas = document.getElementById('footer-ecg-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let w, h;
    function resize() { w = canvas.width = canvas.offsetWidth; h = canvas.height = canvas.offsetHeight; }
    window.addEventListener('resize', resize); resize();

    let offset = 0;
    const green = '#10b981';

    function getEcgY(t) {
        t = ((t % 400) + 400) % 400;
        if (t < 50) return 0;
        if (t < 100) { const p = (t - 50) / 50; return Math.sin(p * Math.PI) * 10; }
        if (t < 120) return 0;
        if (t < 135) { const p = (t - 120) / 15; return -Math.sin(p * Math.PI) * 10; }
        if (t < 160) { const p = (t - 135) / 25; return Math.sin(p * Math.PI) * 80; } // Higher peak
        if (t < 185) { const p = (t - 160) / 25; return -Math.sin(p * Math.PI) * 20; }
        if (t < 210) return 0;
        if (t < 290) { const p = (t - 210) / 80; return Math.sin(p * Math.PI) * 18; }
        return 0;
    }

    function draw() {
        ctx.clearRect(0, 0, w, h);
        const midY = h * 0.7; // Lower baseline within the 120px height
        const scanX = (offset * 3.5) % w;

        // Subtle Gray ECG Grid (Matching Reference)
        // Sub-grid (very faint)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 0.5;
        for (let x = 0; x < w; x += 10) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
        for (let y = 0; y < h; y += 10) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

        // Major-grid (Gray, slightly visible)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
        ctx.lineWidth = 1;
        for (let x = 0; x < w; x += 50) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
        for (let y = 0; y < h; y += 50) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

        // Draw the trace path with smooth phosphor decay (segment-based)
        ctx.lineWidth = 3;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.shadowBlur = 8;
        ctx.shadowColor = green;

        const segmentSize = 4;
        const tailLength = 500;

        for (let x = 0; x < w; x += segmentSize) {
            let dist = scanX - x;
            if (dist < 0) dist += w; // Handle wrap-around distance

            if (dist > tailLength) continue; // Only draw the fading tail

            let alpha = 1 - (dist / tailLength);
            alpha = Math.pow(alpha, 2); // Smooth non-linear decay for realism

            ctx.beginPath();
            ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;
            const y1 = midY - getEcgY(x + offset);
            const y2 = midY - getEcgY(x + segmentSize + offset);
            ctx.moveTo(x, y1);
            ctx.lineTo(x + segmentSize, y2);
            ctx.stroke();
        }

        // Bright Glowing Head
        const headY = midY - getEcgY(scanX + offset);
        ctx.beginPath();
        ctx.fillStyle = '#fff';
        ctx.shadowBlur = 25;
        ctx.shadowColor = '#fff';
        ctx.arc(scanX, headY, 4, 0, Math.PI * 2);
        ctx.fill();

        // Secondary glow
        ctx.beginPath();
        ctx.fillStyle = green;
        ctx.globalAlpha = 0.4;
        ctx.arc(scanX, headY, 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1.0;

        offset += 1.6; // Slower speed (was 1.8)
        requestAnimationFrame(draw);
    }
    draw();
}

// ── NAV ──
function initNavigation() {
    const navHome = document.getElementById('nav-home');
    const navAnalysis = document.getElementById('nav-analysis');
    const navSystem = document.getElementById('nav-system');
    const heroView = document.getElementById('hero-view');
    const resultsView = document.getElementById('results-view');
    const systemView = document.getElementById('system-view');
    const diseaseView = document.getElementById('disease-detail');

    function hideAll() {
        heroView.style.display = 'none';
        resultsView.style.display = 'none';
        if (systemView) systemView.style.display = 'none';
        if (diseaseView) diseaseView.style.display = 'none';
        document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
    }
    function showHome() {
        hideAll(); heroView.style.display = 'flex';
        navHome.classList.add('active'); window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    navHome.addEventListener('click', showHome);
    document.getElementById('back-btn').addEventListener('click', showHome);
    document.getElementById('system-back-btn')?.addEventListener('click', showHome);

    navAnalysis?.addEventListener('click', () => {
        if (!currentResults) return;
        hideAll(); resultsView.style.display = 'block';
        navAnalysis.classList.add('active');
    });

    navSystem?.addEventListener('click', () => {
        hideAll(); if (systemView) systemView.style.display = 'block';
        navSystem.classList.add('active');
        // Check model status
        const statusEl = document.getElementById('system-model-status');
        fetch(`${API_BASE}/health`).then(r => r.json()).then(d => {
            if (statusEl) statusEl.textContent = d.model_loaded ? '✅ Trained model loaded and active.' : '⚠ No trained model. Using random weights — predictions are NOT meaningful.';
            if (statusEl && !d.model_loaded) statusEl.style.color = '#C72C41';
        }).catch(() => { if (statusEl) statusEl.textContent = '✗ Backend not reachable.'; });
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}
function showResultsView() {
    document.getElementById('hero-view').style.display = 'none';
    document.getElementById('results-view').style.display = 'block';
    const sv = document.getElementById('system-view'); if (sv) sv.style.display = 'none';
    const dv = document.getElementById('disease-detail'); if (dv) dv.style.display = 'none';
    document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
    document.getElementById('nav-analysis')?.classList.add('active');
    window.scrollTo({ top: 0 });
}

// ── THEME TOGGLE ──
function initThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    const darkLabel = document.getElementById('theme-dark-label');
    const lightLabel = document.getElementById('theme-light-label');
    const videoDark = document.getElementById('video-dark');
    const videoLight = document.getElementById('video-light');
    if (!toggle) return;

    // Default to dark, but check localStorage
    const savedTheme = localStorage.getItem('cardio-theme') || 'dark';
    isDark = (savedTheme === 'dark');

    let lightVideoPreloaded = false;

    const applyTheme = () => {
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        darkLabel?.classList.toggle('active', isDark);
        lightLabel?.classList.toggle('active', !isDark);

        // Play only the active video, pause the other (saves GPU/CPU)
        if (videoDark && videoLight) {
            if (isDark) {
                videoLight.pause();
                videoDark.play().catch(() => {});
            } else {
                videoDark.pause();
                // Preload color video on first switch
                if (!lightVideoPreloaded) {
                    videoLight.preload = 'auto';
                    videoLight.load();
                    lightVideoPreloaded = true;
                }
                videoLight.play().catch(() => {});
            }
        }

        if (currentResults) {
            renderECGChart(currentResults.waveform, currentResults.markers, currentLead);
        }
    };

    // Apply initial theme
    applyTheme();

    toggle.addEventListener('click', () => {
        isDark = !isDark;
        localStorage.setItem('cardio-theme', isDark ? 'dark' : 'light');
        applyTheme();
    });
}

// ── BG ANIMATION — Single smooth ECG waveform ──
function initHeroBgAnimation() {
    const canvas = document.getElementById('hero-bg-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let w, h;
    function resize() { w = canvas.width = canvas.offsetWidth; h = canvas.height = canvas.offsetHeight; }
    window.addEventListener('resize', resize); resize();
    let offset = 0;

    // Realistic smooth PQRST waveform using bezier-like interpolation
    function ecgWave(t) {
        t = ((t % 600) + 600) % 600; // period = 600px
        // Baseline
        if (t < 80) return 0;
        // P-wave (smooth bump)
        if (t < 140) { const p = (t - 80) / 60; return Math.sin(p * Math.PI) * 18; }
        // PQ segment
        if (t < 170) return 0;
        // Q dip
        if (t < 190) { const p = (t - 170) / 20; return -Math.sin(p * Math.PI) * 12; }
        // R peak (tall sharp spike with smooth rise/fall)
        if (t < 220) { const p = (t - 190) / 30; return Math.sin(p * Math.PI) * 120; }
        // S dip
        if (t < 248) { const p = (t - 220) / 28; return -Math.sin(p * Math.PI) * 25; }
        // ST segment
        if (t < 290) return 0;
        // T-wave (wide smooth bump)
        if (t < 380) { const p = (t - 290) / 90; return Math.sin(p * Math.PI) * 28; }
        // Back to baseline
        return 0;
    }

    function draw() {
        ctx.clearRect(0, 0, w, h);
        const mid = h * 0.58;

        // Read theme colors from CSS
        const style = getComputedStyle(document.documentElement);
        const glowColor = style.getPropertyValue('--ecg-glow').trim() || 'rgba(0,0,0,0.08)';
        const strokeColor = style.getPropertyValue('--ecg-stroke').trim() || 'rgba(0,0,0,0.35)';

        // Build smooth path points
        const points = [];
        for (let x = 0; x <= w; x += 1) {
            const y = mid - ecgWave(x + offset);
            points.push({ x, y });
        }

        // Draw subtle glow underneath
        ctx.beginPath();
        ctx.strokeStyle = glowColor;
        ctx.lineWidth = 12;
        ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.stroke();

        // Draw main ECG line — single, clean, bold
        ctx.beginPath();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.stroke();

        offset += 1.2;
        requestAnimationFrame(draw);
    }
    draw();
}

// ── LIVE FEED ──
function initLiveAIFeed() {
    const el = document.getElementById('hero-ai-status'); if (!el) return;
    const msgs = ['Neural engine ready. Awaiting ECG input.', 'PTB-XL dataset loaded. 21,837 records indexed.', 'CNN-BiLSTM model warm. Ready for inference.', 'Upload .HEA + .DAT to begin classification.'];
    let i = 0; el.style.transition = 'opacity 0.4s ease';
    setInterval(() => { i = (i + 1) % msgs.length; el.style.opacity = '0'; setTimeout(() => { el.textContent = msgs[i]; el.style.opacity = '1'; }, 400); }, 4000);
}

// ── UPLOAD ──
function initUpload() {
    const uploadBtn = document.getElementById('upload-btn');
    const fileInputHea = document.getElementById('file-input-hea');
    const fileInputDat = document.getElementById('file-input-dat');
    const statusEl = document.getElementById('upload-file-status');
    const labelEl = document.getElementById('upload-cta-label');

    uploadBtn.addEventListener('click', () => {
        if (!heaFile) { fileInputHea.value = ''; fileInputHea.click(); }
        else { fileInputDat.value = ''; fileInputDat.click(); }
    });
    fileInputHea.addEventListener('change', (e) => {
        if (e.target.files.length) {
            heaFile = e.target.files[0];
            if (statusEl) { statusEl.style.color = '#10b981'; statusEl.textContent = `✓ ${heaFile.name} selected`; }
            if (labelEl) labelEl.textContent = 'SELECT .DAT';
            setTimeout(() => { fileInputDat.value = ''; fileInputDat.click(); }, 300);
        }
    });
    fileInputDat.addEventListener('change', (e) => {
        if (e.target.files.length) {
            datFile = e.target.files[0];
            if (statusEl) { statusEl.style.color = '#10b981'; statusEl.textContent = `✓ ${heaFile.name} + ${datFile.name}`; }
            if (heaFile && datFile) analyzeUploadedFiles();
        }
    });
}

function resetUploadUI() {
    heaFile = null; datFile = null;
    const s = document.getElementById('upload-file-status');
    const l = document.getElementById('upload-cta-label');
    if (s) { s.textContent = ''; s.style.color = '#10b981'; }
    if (l) l.textContent = 'SELECT .HEA';
}

function showLoader(text) { const o = document.getElementById('loader-overlay'); o.classList.remove('hidden'); o.querySelector('.loader-text').textContent = text || 'PROCESSING...'; }
function hideLoader() { document.getElementById('loader-overlay').classList.add('hidden'); }

function showError(msg) {
    hideLoader();
    const statusEl = document.getElementById('upload-file-status');
    if (statusEl) { statusEl.style.color = '#C72C41'; statusEl.textContent = `✗ ERROR: ${msg}`; }
    setTimeout(() => alert(`Analysis Error: ${msg}`), 200);
}

async function analyzeUploadedFiles() {
    showLoader('ANALYZING ECG RECORDING...');
    const recordName = heaFile ? heaFile.name.replace('.hea', '') : 'UNKNOWN';
    try {
        const fd = new FormData();
        fd.append('hea_file', heaFile); fd.append('dat_file', datFile);
        const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: fd });
        if (!res.ok) { let em = 'Analysis failed'; try { em = (await res.json()).error || em; } catch (_) { } throw new Error(em); }
        const data = await res.json();
        hideLoader(); resetUploadUI(); displayResults(data, recordName);
    } catch (err) { console.error('Analysis error:', err); showError(err.message); resetUploadUI(); }
}

// ── RESULTS ──
function displayResults(results, recordId) {
    currentResults = results; currentRecordId = recordId.toUpperCase(); showResultsView();
    document.getElementById('record-id').textContent = currentRecordId;
    document.getElementById('results-timestamp').textContent = new Date().toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    renderDiagnosis(results.prediction); renderMetrics(results.metrics);
    currentLead = 'II'; initLeadSelector();
    renderECGChart(results.waveform, results.markers, currentLead);
}

function renderDiagnosis(pred) {
    const cn = pred.class_name, isNormal = cn === 'Normal ECG';
    const titleEl = document.getElementById('diag-title');
    titleEl.textContent = CLASS_SHORT[cn] || cn.toUpperCase();
    titleEl.classList.toggle('critical', !isNormal);
    // Make title clickable for disease detail
    titleEl.classList.add('clickable');
    titleEl.onclick = () => showDiseaseDetail(cn);
    document.getElementById('diag-banner').style.background = isNormal ? '#000' : '#1A1A1A';
    animateValue('diag-conf', 0, Math.round(pred.confidence), 1200, v => `${v}%`);
    const probList = document.getElementById('prob-list'); probList.innerHTML = '';
    Object.entries(pred.probabilities).forEach(([name, prob]) => {
        const card = document.createElement('div');
        card.className = `prob-card ${prob > 50 && name !== 'Normal ECG' ? 'danger' : ''}`;
        card.innerHTML = `<div class="prob-card-name">${CLASS_SHORT[name] || name}</div><div class="prob-card-value">${prob.toFixed(1)}%</div><div class="prob-card-bar"><div class="prob-card-bar-fill"></div></div>`;
        probList.appendChild(card);
        requestAnimationFrame(() => requestAnimationFrame(() => { card.querySelector('.prob-card-bar-fill').style.width = `${prob}%`; }));
    });
    const expEl = document.getElementById('human-explanation');
    expEl.textContent = isNormal ? 'The rhythm is regular and morphologically sound. No significant anomalies detected in the 12-lead analysis.' : `Structural deviations detected indicative of ${cn}. Review annotated markers in the ECG trace.`;
    expEl.classList.toggle('critical', !isNormal);
}

// ══════════════════════════════════════════════════════════════
// DISEASE DETAIL VIEW
// ══════════════════════════════════════════════════════════════
function showDiseaseDetail(className) {
    const data = DISEASE_DATA[className];
    if (!data) return;
    // Hide results, show disease detail
    document.getElementById('results-view').style.display = 'none';
    document.getElementById('disease-detail').style.display = 'block';
    renderDiseaseView(data, className);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    // Wire back button
    document.getElementById('disease-back-btn').onclick = () => {
        document.getElementById('disease-detail').style.display = 'none';
        document.getElementById('results-view').style.display = 'block';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
}

function renderDiseaseView(data, className) {
    const isNormal = className === 'Normal ECG';
    // Hero
    const titleEl = document.getElementById('disease-title');
    titleEl.innerHTML = data.fullName.replace('\n', '<br>');
    titleEl.classList.toggle('normal', isNormal);
    document.getElementById('disease-subtitle').textContent = data.subtitle;
    document.getElementById('disease-image').src = data.image;
    document.getElementById('disease-badge').textContent = data.shortName + ' — CONDITION INFO';
    // Overview
    document.getElementById('disease-overview').textContent = data.overview;
    // Causes
    const causesList = document.getElementById('disease-causes');
    causesList.innerHTML = data.causes.map(c => `<li>${c}</li>`).join('');
    // Symptoms
    const symptomsList = document.getElementById('disease-symptoms');
    symptomsList.innerHTML = data.symptoms.map(s => `<li>${s}</li>`).join('');
    // ECG Signs
    const ecgList = document.getElementById('disease-ecg-signs');
    ecgList.innerHTML = data.ecgSigns.map(e => `<li>${e}</li>`).join('');
    // Treatment
    const treatList = document.getElementById('disease-treatment');
    treatList.innerHTML = data.treatment.map(t => `<li>${t}</li>`).join('');
    // Clinical Note
    const noteEl = document.getElementById('disease-clinical-note');
    noteEl.textContent = data.clinicalNote;
    noteEl.classList.toggle('critical', !isNormal);
}

function renderMetrics(m) {
    animateValue('metric-hr', 0, Math.round(m.heart_rate || 0), 1000);
    animateValue('metric-pr', 0, Math.round(m.pr_interval || 0), 1000);
    animateValue('metric-qrs', 0, Math.round(m.qrs_duration || 0), 1000);
    animateValue('metric-rr', 0, Math.round(m.rr_interval || 0), 1000);
}

function animateValue(id, start, end, dur, fmt) {
    const el = document.getElementById(id); if (!el) return;
    const t0 = performance.now();
    (function tick(t) { const p = Math.min((t - t0) / dur, 1); const v = Math.round(start + (end - start) * (1 - Math.pow(1 - p, 3))); el.textContent = fmt ? fmt(v) : v; if (p < 1) requestAnimationFrame(tick); })(t0);
}

// ══════════════════════════════════════════════════════════════
// ECG CHART — with all markers + auto-trim padded data
// ══════════════════════════════════════════════════════════════
function initLeadSelector() {
    const c = document.getElementById('lead-selector'); c.innerHTML = '';
    LEAD_NAMES.forEach(lead => {
        const btn = document.createElement('button');
        btn.className = `lead-btn ${lead === currentLead ? 'active' : ''}`;
        btn.textContent = lead;
        btn.addEventListener('click', () => {
            document.querySelectorAll('.lead-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active'); currentLead = lead;
            const lbl = document.getElementById('current-lead-label'); if (lbl) lbl.textContent = lead;
            renderECGChart(currentResults.waveform, currentResults.markers, lead);
        });
        c.appendChild(btn);
    });
}

function getActualLength(data, hintLen) {
    // Use backend hint if available, else detect trailing zeros
    if (hintLen && hintLen < data.length) return hintLen;
    for (let i = data.length - 1; i > 100; i--) {
        if (Math.abs(data[i]) > 0.001) return Math.min(i + 50, data.length);
    }
    return data.length;
}

function renderECGChart(waveform, markers, leadName) {
    const ctx = document.getElementById('ecg-main-chart');
    if (ecgChart) ecgChart.destroy();

    const fullData = waveform.leads[leadName] || [];
    const fullTime = waveform.time || [];
    const actualLen = getActualLength(fullData, waveform.actual_samples);
    const leadData = fullData.slice(0, actualLen);
    const timeData = fullTime.slice(0, actualLen);

    // No downsampling to maintain true R-peak amplitude and sharpness
    const step = 1;
    const sTime = timeData;
    const sData = leadData;

    const chartFg = isDark ? '#F2F0ED' : '#000';
    const datasets = [{
        label: `Lead ${leadName}`, data: sData, borderColor: chartFg,
        borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0, order: 5,
        pointStyle: 'line'
    }];

    const anomalyList = document.getElementById('anomaly-list'); anomalyList.innerHTML = '';
    const mapIdx = (orig) => orig; // 1:1 mapping

    // P-Waves
    if (markers?.p_waves?.length) {
        const d = new Array(sData.length).fill(null);
        markers.p_waves.forEach(i => { if (i < actualLen) { const si = mapIdx(i); if (si >= 0) d[si] = sData[si]; } });
        datasets.push({ label: 'P-Waves', data: d, borderColor: MARKER_COLORS.pWave, backgroundColor: MARKER_COLORS.pWave, pointRadius: 2.5, pointStyle: 'circle', showLine: false, order: 1 });
        anomalyList.innerHTML += `<li class="anomaly-item"><span class="anomaly-icon" style="background:${MARKER_COLORS.pWave}"></span><span><strong>${markers.p_waves.length}</strong> P-Waves identified.</span></li>`;
    }
    // R-Peaks
    if (markers?.r_peaks?.length) {
        const d = new Array(sData.length).fill(null);
        markers.r_peaks.forEach(i => { if (i < actualLen) { const si = mapIdx(i); if (si >= 0) d[si] = sData[si]; } });
        datasets.push({ label: 'R-Peaks', data: d, borderColor: MARKER_COLORS.rPeak, backgroundColor: MARKER_COLORS.rPeak, pointRadius: 3, pointStyle: 'circle', showLine: false, order: 2 });
        anomalyList.innerHTML += `<li class="anomaly-item"><span class="anomaly-icon" style="background:${MARKER_COLORS.rPeak}"></span><span><strong>${markers.r_peaks.length}</strong> R-Peaks detected.</span></li>`;
    }
    // T-Waves
    if (markers?.t_waves?.length) {
        const d = new Array(sData.length).fill(null);
        markers.t_waves.forEach(i => { if (i < actualLen) { const si = mapIdx(i); if (si >= 0) d[si] = sData[si]; } });
        datasets.push({ label: 'T-Waves', data: d, borderColor: MARKER_COLORS.tWave, backgroundColor: MARKER_COLORS.tWave, pointRadius: 2.5, pointStyle: 'circle', showLine: false, order: 3 });
        anomalyList.innerHTML += `<li class="anomaly-item"><span class="anomaly-icon" style="background:${MARKER_COLORS.tWave}"></span><span><strong>${markers.t_waves.length}</strong> T-Waves identified.</span></li>`;
    }
    // QRS
    if (markers?.qrs_complexes?.length) {
        const d = new Array(sData.length).fill(null);
        markers.qrs_complexes.forEach(q => { const pk = q.peak || q.onset; if (pk < actualLen) { const si = mapIdx(pk); if (si >= 0) d[si] = sData[si]; } });
        datasets.push({ label: 'QRS Complex', data: d, borderColor: MARKER_COLORS.qrs, backgroundColor: MARKER_COLORS.qrs, pointRadius: 2.5, pointStyle: 'circle', showLine: false, order: 4 });
        anomalyList.innerHTML += `<li class="anomaly-item"><span class="anomaly-icon" style="background:${MARKER_COLORS.qrs}"></span><span><strong>${markers.qrs_complexes.length}</strong> QRS Complexes.</span></li>`;
    }

    const isCrit = currentResults.prediction.class_name !== 'Normal ECG';
    anomalyList.innerHTML += isCrit
        ? `<li class="anomaly-item"><span class="anomaly-icon danger"></span><span style="color:#C72C41;font-weight:700">Structural deviation detected.</span></li>`
        : `<li class="anomaly-item"><span class="anomaly-icon"></span><span>All intervals within normal parameters.</span></li>`;

    // Read theme colors for chart
    const style = getComputedStyle(document.documentElement);
    const fgColor = style.getPropertyValue('--fg').trim() || '#000';
    const fgMuted = style.getPropertyValue('--fg-muted').trim() || '#8A8886';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.04)';
    const tooltipBg = isDark ? '#222' : '#000';

    ecgChart = new Chart(ctx, {
        type: 'line', data: { labels: sTime.map(t => t.toFixed(2)), datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1000, easing: 'easeOutQuart' },
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: true, position: 'top', labels: { color: fgColor, font: { family: "'JetBrains Mono'", size: 10 }, usePointStyle: true, padding: 16 } },
                tooltip: { backgroundColor: tooltipBg, titleColor: '#fff', bodyColor: '#ccc', cornerRadius: 0, padding: 10, bodyFont: { family: "'JetBrains Mono'", size: 10 } },
            },
            scales: {
                x: { title: { display: true, text: 'Time (s)', font: { family: "'JetBrains Mono'", size: 10, weight: 'bold' }, color: fgMuted }, ticks: { color: fgMuted, font: { family: "'JetBrains Mono'", size: 9 }, maxTicksLimit: 12 }, grid: { color: gridColor } },
                y: { title: { display: true, text: 'mV', font: { family: "'JetBrains Mono'", size: 10, weight: 'bold' }, color: fgMuted }, ticks: { color: fgMuted, font: { family: "'JetBrains Mono'", size: 9 } }, grid: { color: gridColor } },
            },
        },
    });
}

// ══════════════════════════════════════════════════════════════
// DOWNLOAD REPORT — Visual PNG with all 12 ECG leads + diagnosis
// ══════════════════════════════════════════════════════════════
function downloadReport() {
    if (!currentResults) return;
    const r = currentResults, wf = r.waveform, mk = r.markers, p = r.prediction, m = r.metrics;

    const W = 2400, leadH = 160, headerH = 280, footerH = 100, gap = 4;
    const totalH = headerH + 12 * (leadH + gap) + footerH;
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = totalH;
    const ctx = canvas.getContext('2d');

    // White background
    ctx.fillStyle = '#FFF'; ctx.fillRect(0, 0, W, totalH);

    // ── Header (black bar) ──
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, headerH);
    ctx.fillStyle = '#FFF'; ctx.font = 'bold 36px Inter, sans-serif';
    ctx.fillText('CARDIO.AI — ECG DIAGNOSTIC REPORT', 40, 50);
    ctx.font = '15px monospace'; ctx.fillStyle = '#AAA';
    ctx.fillText(`Record: ${currentRecordId}  |  ${new Date().toLocaleString()}  |  v2.0`, 40, 78);

    // Diagnosis badge
    const isNorm = p.class_name === 'Normal ECG';
    ctx.fillStyle = isNorm ? '#10b981' : '#D72638';
    ctx.fillRect(40, 100, 320, 44);
    ctx.fillStyle = '#FFF'; ctx.font = 'bold 20px Inter, sans-serif';
    ctx.fillText(p.class_name, 52, 130);
    ctx.fillStyle = '#CCC'; ctx.font = '15px monospace';
    ctx.fillText(`Confidence: ${p.confidence}%`, 380, 130);

    // Metrics
    const mets = [['HR', `${Math.round(m.heart_rate || 0)} BPM`], ['PR', `${Math.round(m.pr_interval || 0)} ms`], ['QRS', `${Math.round(m.qrs_duration || 0)} ms`], ['RR', `${Math.round(m.rr_interval || 0)} ms`], ['R-Peaks', `${mk.r_peaks?.length || 0}`], ['P-Waves', `${mk.p_waves?.length || 0}`], ['T-Waves', `${mk.t_waves?.length || 0}`], ['QRS#', `${mk.qrs_complexes?.length || 0}`]];
    const mW = (W - 80) / mets.length;
    mets.forEach(([label, val], i) => {
        ctx.fillStyle = '#777'; ctx.font = '11px monospace'; ctx.fillText(label.toUpperCase(), 40 + i * mW, 178);
        ctx.fillStyle = '#FFF'; ctx.font = 'bold 18px Inter, sans-serif'; ctx.fillText(val, 40 + i * mW, 200);
    });

    // Probabilities bar
    let px = 40;
    Object.entries(p.probabilities).forEach(([name, prob]) => {
        const bw = 130;
        ctx.fillStyle = '#333'; ctx.fillRect(px, 224, bw, 12);
        ctx.fillStyle = prob > 50 && name !== 'Normal ECG' ? '#D72638' : '#10b981';
        ctx.fillRect(px, 224, bw * prob / 100, 12);
        ctx.fillStyle = '#999'; ctx.font = '10px monospace';
        ctx.fillText(`${CLASS_SHORT[name] || name} ${prob.toFixed(1)}%`, px, 252);
        px += bw + 30;
    });

    // ── 12-Lead ECG Traces ──
    const actualSamples = wf.actual_samples || 5000;
    LEAD_NAMES.forEach((leadName, idx) => {
        const yTop = headerH + idx * (leadH + gap);
        const fullData = wf.leads[leadName] || [];
        const trimLen = getActualLength(fullData, actualSamples);
        const data = fullData.slice(0, trimLen);
        if (!data.length) return;

        // Alternating row bg
        ctx.fillStyle = idx % 2 === 0 ? '#FAFAFA' : '#F4F4F4';
        ctx.fillRect(0, yTop, W, leadH);

        // Lead label
        ctx.fillStyle = '#000'; ctx.font = 'bold 14px monospace'; ctx.fillText(leadName, 12, yTop + 20);

        // Grid
        ctx.strokeStyle = 'rgba(0,0,0,0.06)'; ctx.lineWidth = 0.5;
        for (let g = 0; g <= 10; g++) { const gx = 60 + g * ((W - 80) / 10); ctx.beginPath(); ctx.moveTo(gx, yTop); ctx.lineTo(gx, yTop + leadH); ctx.stroke(); }

        // Scale
        let dMin = Infinity, dMax = -Infinity;
        for (const v of data) { if (v < dMin) dMin = v; if (v > dMax) dMax = v; }
        const dRange = (dMax - dMin) || 1, padY = 15, plotH = leadH - padY * 2, plotW = W - 80;

        // Trace
        ctx.beginPath(); ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
        for (let i = 0; i < data.length; i++) {
            const x = 60 + (i / data.length) * plotW;
            const y = yTop + padY + plotH - ((data[i] - dMin) / dRange) * plotH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Markers
        function dm(positions, color, shape) {
            if (!positions?.length) return;
            positions.forEach(pos => {
                const ri = typeof pos === 'object' ? (pos.peak || pos.onset) : pos;
                if (ri >= trimLen || ri >= data.length) return;
                const x = 60 + (ri / data.length) * plotW;
                const y = yTop + padY + plotH - ((data[ri] - dMin) / dRange) * plotH;
                ctx.fillStyle = color;
                if (shape === 'c') { ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); }
                else if (shape === 't') { ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x - 4, y + 3); ctx.lineTo(x + 4, y + 3); ctx.closePath(); ctx.fill(); }
                else if (shape === 's') { ctx.fillRect(x - 3, y - 3, 6, 6); }
                else if (shape === 'd') { ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x + 3, y); ctx.lineTo(x, y + 4); ctx.lineTo(x - 3, y); ctx.closePath(); ctx.fill(); }
            });
        }
        dm(mk.p_waves, '#10b981', 'c');
        dm(mk.r_peaks, '#f59e0b', 't');
        dm(mk.t_waves, '#3b82f6', 's');
        dm(mk.qrs_complexes, '#ef4444', 'd');

        // Separator
        ctx.strokeStyle = 'rgba(0,0,0,0.12)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, yTop + leadH); ctx.lineTo(W, yTop + leadH); ctx.stroke();
    });

    // ── Footer ──
    const fy = headerH + 12 * (leadH + gap) + 15;
    ctx.fillStyle = '#F0F0F0'; ctx.fillRect(0, fy - 15, W, footerH);
    ctx.font = '13px monospace';
    [['● P-Wave', '#10b981'], ['▲ R-Peak', '#f59e0b'], ['■ T-Wave', '#3b82f6'], ['◆ QRS', '#ef4444']].forEach(([t, c], i) => {
        ctx.fillStyle = c; ctx.fillText(t, 40 + i * 180, fy + 12);
    });
    ctx.fillStyle = '#999'; ctx.font = '11px monospace';
    ctx.fillText('Generated by CARDIO.AI for research purposes. Consult a cardiologist for clinical interpretation.', 40, fy + 50);

    // Download
    canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `CardioAI_Report_${currentRecordId.replace(/\s+/g, '_')}.png`;
        a.click(); URL.revokeObjectURL(url);
    }, 'image/png');
}
