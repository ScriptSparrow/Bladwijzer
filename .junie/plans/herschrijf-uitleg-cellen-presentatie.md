---
sessionId: session-260611-232007-1i4w
---

# Doel & Aanpak

### Doel
De markdown-uitlegcellen in `notebooks/bladwijzer.ipynb` herschrijven zodat ze geschikt zijn als presentatiemateriaal voor docenten die **niet diep technisch** zijn, terwijl de notebook ook bruikbaar blijft als technisch verslag.

### Publiek & toon
- Doelgroep: docenten zonder ML-achtergrond.
- Begrippen zoals *bounding box*, *mAP*, *IoU*, *augmentatie*, *recall/precision*, *stratified split* worden **eerst in gewone taal** uitgelegd, daarna pas technisch ingevuld.
- Nederlands, korte zinnen, geen jargon zonder definitie.

### Vaste structuur per uitlegcel
Elke markdown-cel krijgt drie vaste kopjes (h3):
1. **Wat doen we?** — één à twee zinnen, in klare taal.
2. **Waarom doen we het?** — de motivatie, gekoppeld aan het projectdoel (invasieve berenklauw detecteren, min false positives).
3. **Hoe doen we het?** — de technische uitvoering, inclusief de relevante parameters/keuzes.

Afsluitend onder elke cel een blokje **🎤 Tijdens de demo zeg ik** met 2–4 bullets als spreekkaartje.

### Niet in scope
- Code-cellen blijven ongewijzigd.
- Geen nieuwe analyses, plots of metrics toevoegen.
- Geen extra intro-slide, mermaid-diagram of conclusie-cel (niet gekozen door de gebruiker).
- Geen wijziging aan datasets, training of modelbestanden.

# Cellen-inventaris

### Bestand
`notebooks/bladwijzer.ipynb` — 11 markdown-cellen op de volgende indices:

 Index | Sectie | Onderwerp |
---|---|---|
 0 | Titel | Project intro, doel, dataset, pipeline, test-set waarschuwing |
 1 | 1 | Imports & config |
 3 | 2 | Dataset statistieken |
 5 | 3 | Sanity check / visualisatie bounding boxes |
 9 | 4 | Training (YOLO11n keuzes, augmentatie) |
 13 | 4b | Trainingsscherm / results.csv tabel |
 15 | 5 | Validatie val-set, per klasse, recall/precision/mAP |
 18 | (vervolg 5) | Confusion matrix interpretatie |
 19 | 6 | Test-set eindcijfer |
 21 | 7 | Beste model bewaren |
 23 | 8 | Inference demo |

Elke cel wordt herschreven volgens de vaste *Wat / Waarom / Hoe + spreekkaart* structuur. De bestaande inhoudelijke punten blijven behouden — alleen de **vorm, volgorde en toegankelijkheid** veranderen.

### Begrippenlijst die in gewone taal wordt geïntroduceerd op de eerste plek waar ze voorkomen
- **Bounding box** (cel 5): "een rechthoek om het object op de foto".
- **Class-id / klasse** (cel 5).
- **Augmentatie, mosaic, mixup, hsv** (cel 9): "trucjes om van één foto meerdere trainfoto's te maken".
- **Epoch, early stopping, patience** (cel 9).
- **Precision, Recall, mAP50, mAP50-95, IoU** (cel 15): kort intuïtief + formule-loze uitleg.
- **Confusion matrix** (cel 18).
- **Confidence threshold** (cel 23).

# Voorbeeld-herschrijving

### Voorbeeld: cel 15 (Validatie val-set)

```markdown
## 5. Validatie op de val-set — hoe goed is het model echt?

### Wat doen we?
We laten het getrainde model voorspellen op de **validatieset**: foto's die het tijdens trainen nog nooit heeft gezien. Vervolgens vergelijken we de voorspellingen met de echte labels.

### Waarom doen we het?
Omdat ons doel is om **berenklauw zo weinig mogelijk te missen** (invasieve plant!), is een gemiddeld cijfer over alle foto's misleidend: madeliefjes komen ~10× vaker voor, dus een model dat berenklauw bijna mist scoort alsnog hoog. Daarom kijken we **per klasse** en specifiek naar twee maten:

- **Recall** — *Van alle echte berenklauwen op de foto's: hoeveel heeft het model gevonden?* Hoge recall = weinig gemist.
- **Precision** — *Van alles wat het model 'berenklauw' noemde: hoeveel klopte er echt?* Hoge precision = weinig vals alarm.

Voor onze use-case is **recall belangrijker** dan precision: een gemiste invasieve plant kost meer dan een vals alarm.

### Hoe doen we het?
- We roepen `model.val(split="val")` aan; Ultralytics rekent automatisch precision, recall en mAP per klasse uit.
- **mAP50** = gemiddelde score bij 'box overlapt ≥ 50% met het echte label'. Geeft een gebalanceerd beeld.
- **mAP50-95** is strenger en straft slordig getekende boxes af.
- Op basis hiervan beslissen we of we nog moeten bijsturen — maar **nooit** door naar de test-set te kijken.

> 🎤 **Tijdens de demo zeg ik:**
> - "Validatieset = oefententamen, testset = echte tentamen."
> - "Let vooral op de berenklauw-rij: recall is hier ons belangrijkste cijfer."
> - "Een hoge gemiddelde score zegt weinig zolang de minderheidsklasse achterblijft."
```

Dezelfde sjabloon wordt toegepast op alle 11 cellen, met inhoudelijk behoud van de bestaande argumenten (bv. de waarschuwing rond test-set leakage in cel 0/19, de keuzes voor `yolo11n.pt` / `mosaic` / `mixup` in cel 9, de interpretatie van de confusion matrix in cel 18, enz.).

# Delivery Steps

###   Step 1: Intro- en setup-cellen herschrijven (cellen 0, 1)
De titel- en imports-uitleg presentatie-klaar maken.

- Cel 0 (titel): herschrijven met klare-taal intro over het probleem (invasieve berenklauw vs onschuldig madeliefje), het doel (zo min mogelijk gemiste berenklauw), de dataset (~156 foto's, 80/10/10 split) en de pipeline op één regel. Test-set-waarschuwing in een aparte callout. Spreekkaart-bullets onderaan.
- Cel 1 (Imports & config): omzetten naar Wat / Waarom / Hoe — Wat: alle instellingen centraal; Waarom: reproduceerbaar + onderhoudbaar; Hoe: `SEED`, `ROOT`, `NAMES`, `COLORS`. Spreekkaart toevoegen.

###   Step 2: Data-inspectie cellen herschrijven (cellen 3, 5)
Uitleg rond statistieken en visuele sanity check toegankelijk maken.

- Cel 3 (Dataset statistieken): Wat/Waarom/Hoe-structuur; uitleg waarom we *boxes* tellen i.p.v. images, en waarom de 10:1 ratio er toe doet. Spreekkaart met de kernboodschap 'cijfers eerst checken, niet blind trainen'.
- Cel 5 (Sanity check): introduceer 'bounding box' in gewone taal als 'rechthoek om het object'. Leg uit waarom we visueel checken (coördinatenfouten, class-id mix-up). Spreekkaart met 1 zin over wat docenten in de plot moeten zien.

###   Step 3: Trainings-cellen herschrijven (cellen 9, 13)
De trainings-uitleg ontdoen van jargon en geschikt maken voor docenten.

- Cel 9 (Training): introduceer in klare taal: *pretrained model*, *epoch*, *early stopping/patience*, *augmentatie* (mosaic/mixup/hsv) met een korte alledaagse analogie ('alsof je het model dezelfde foto in tien verschillende belichtingen laat zien'). Behoud alle bestaande keuzes en hun motivatie in het 'Hoe'-blok. Spreekkaart met '3 dingen om over de training te zeggen'.
- Cel 13 (Trainingsscherm 4b): Wat/Waarom/Hoe rond de results.csv-tabel; uitleg dat de beste epoch op `mAP50-95` wordt gekozen. Spreekkaart.

###   Step 4: Evaluatie-cellen herschrijven (cellen 15, 18)
De val-set evaluatie en confusion matrix begrijpelijk maken voor niet-technici.

- Cel 15 (Validatie): volledig herschreven volgens het sjabloon uit het voorbeeld in de proposal — recall en precision in alledaagse taal, mAP50 vs mAP50-95 in één zin, link naar projectdoel (recall > precision voor berenklauw). Spreekkaart.
- Cel 18 (Confusion matrix): Wat/Waarom/Hoe met heel concrete voorbeelden per kwadrant ('berenklauw → background = gemist') en welke actie daarbij hoort. Spreekkaart die uitlegt hoe je de matrix in 10 seconden leest.

###   Step 5: Test-, opslag- en demo-cellen herschrijven (cellen 19, 21, 23)
De afsluitende cellen presentatie-klaar maken zodat de demo sterk eindigt.

- Cel 19 (Test-set): Wat/Waarom/Hoe met nadrukkelijke 'niet meer tunen na deze cel'-callout in klare taal; uitleg waarom test < val normaal is. Spreekkaart met 2 zinnen die de eerlijkheid van het cijfer benadrukken.
- Cel 21 (Beste model bewaren): Wat/Waarom/Hoe rond `best.pt` → `models/bladwijzer_best.pt`; uitleg over scheiding R&D vs deliverable in alledaagse taal ('werkmap vs eindversie'). Spreekkaart.
- Cel 23 (Inference demo): Wat/Waarom/Hoe rond `conf=0.25` en `predict`; introduceer 'confidence threshold' in klare taal ('hoe zeker moet het model zijn voordat we het tonen'). Spreekkaart met afsluiter voor de presentatie.