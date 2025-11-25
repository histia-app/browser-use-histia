# API FastAPI : Agents Histia d'extraction de données

Ce répertoire contient une API FastAPI complète pour utiliser les agents d'extraction Histia. Chaque agent dispose de son propre endpoint POST dédié avec documentation complète et exemples d'utilisation.

## Prérequis

- Python 3.11+
- Dépendances installées (FastAPI, Uvicorn et les dépendances du projet) via `uv sync` ou `pip install -e .[dev]`
- Variables d'environnement pour le backend LLM :
  - `BROWSER_USE_API_KEY` (recommandé) pour utiliser ChatBrowserUse
  - `OPENAI_API_KEY` et éventuellement `OPENAI_API_BASE` pour un backend LiteLLM/OpenAI
  - ou `LLM_BACKEND=gemini` + `GOOGLE_API_KEY` (ou `GEMINI_API_KEY`) pour basculer sur Gemini

## Démarrage

Lancez l'API en mode développement avec rechargement à chaud :

```bash
uvicorn examples.histia.fastapi_agents:app --reload
```

L'explorateur interactif est disponible sur `http://localhost:8000/docs` (OpenAPI/Swagger) et `http://localhost:8000/redoc`.

## Agents disponibles

Les agents sont organisés en deux catégories :

1. **Agents généraux** : Agents polyvalents qui fonctionnent avec de multiples sources et structures de sites web
2. **Agents spécialisés** : Agents optimisés pour des plateformes spécifiques, offrant une meilleure performance et fiabilité pour leurs plateformes cibles

### Guide de sélection rapide

| Site/Plateforme | Agent recommandé | Alternative |
|----------------|------------------|-------------|
| **Product Hunt** (leaderboard) | `product_hunt_leaderboard` | `startup_listing` |
| **FutureTools** | `futuretools_extractor` | `universal_startup_extractor` |
| **AppSumo** (What's hot) | `appsumo_hot_extractor` | - |
| **AppSumo** (New arrivals) | `appsumo_new_extractor` | - |
| **BetaList** | `betalist_extractor` | `startup_listing` |
| **Station F** | `stationf_companies_extractor` | `universal_startup_extractor` |
| **Zone Secure** | `zone_secure_startups_extractor` | `universal_startup_extractor` |
| **Airtable** | `airtable_extractor` | - |
| **Sites personnalisés/inconnus** | `universal_startup_extractor` | `startup_listing` |
| **Page produit individuelle** | `product_research` | - |
| **Liste de startups générique** | `startup_listing` | `universal_startup_extractor` |

> 💡 **Recommandation** : Pour les plateformes listées ci-dessus, utilisez toujours l'agent spécialisé pour de meilleures performances. Les agents spécialisés sont généralement 2-3x plus rapides et plus fiables que les agents généraux.

---

## Agents généraux

Ces agents peuvent fonctionner avec de nombreux sites web différents. Ils utilisent des stratégies guidées par LLM pour s'adapter à différentes structures de pages.

### 1. Product Research (`product_research`) (PAS LISTING)

Agent de recherche de produits qui extrait des informations structurées sur les entreprises et leurs produits depuis des listings (Product Hunt, BetaList, etc.). Retourne un profil d'entreprise complet, les produits principaux, et des faits notables.

**Endpoint :** `POST /agents/product_research/run`

**Paramètres :**
- `url` (requis, `AnyHttpUrl`) : URL du listing à analyser (Product Hunt, BetaList, etc.)
- `max_products` (optionnel, `int`, défaut: `3`) : Nombre maximum de produits ou variantes à résumer (min: 1, max: 10)
- `output_path` (optionnel, `Path`, défaut: `"product_research_report.json"`) : Chemin de destination pour le rapport JSON généré

**Exemple complet avec tous les paramètres :**

```bash
curl -X POST "http://localhost:8000/agents/product_research/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.producthunt.com/posts/example-product",
       "max_products": 5,
       "output_path": "my_product_research.json"
     }'
```

**Exemple pour BetaList :**

```bash
curl -X POST "http://localhost:8000/agents/product_research/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://betalist.com/startups/example-startup",
       "max_products": 3,
       "output_path": "betalist_research.json"
     }'
```

**Exemple pour extraire le maximum de produits (10 produits) :**

```bash
curl -X POST "http://localhost:8000/agents/product_research/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.producthunt.com/posts/complex-product",
       "max_products": 10,
       "output_path": "full_product_research.json"
     }'
```

**Exemple minimal (seulement l'URL requise) :**

```bash
curl -X POST "http://localhost:8000/agents/product_research/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.producthunt.com/posts/my-awesome-product"
     }'
```

**Exemple avec Python :**

```python
import requests

# Exemple complet avec analyse des résultats
response = requests.post(
    "http://localhost:8000/agents/product_research/run",
    json={
        "url": "https://www.producthunt.com/posts/example-product",
        "max_products": 5,
        "output_path": "research_report.json"
    }
)
report = response.json()

# Afficher les informations de l'entreprise
company = report['company']
print(f"Entreprise: {company['name']}")
print(f"Site web: {company.get('official_website', 'N/A')}")
print(f"LinkedIn: {company.get('linkedin_page', 'N/A')}")
if company.get('other_facts'):
    print(f"Faits notables: {', '.join(company['other_facts'][:3])}")

# Afficher les produits
print(f"\nProduits trouvés: {len(report['products'])}")
for i, product in enumerate(report['products'], 1):
    print(f"\n{i}. {product['product_name']}")
    print(f"   Description: {product['what_it_does']}")
    print(f"   Modèle: {product.get('go_to_market', 'N/A')}")
    print(f"   Audience: {product.get('target_audience', 'N/A')}")

# Exemple minimal
response = requests.post(
    "http://localhost:8000/agents/product_research/run",
    json={
        "url": "https://betalist.com/startups/my-startup"
    }
)
```

**Note importante :** Cet agent analyse une page de listing spécifique (pas une liste de produits). Il extrait les informations détaillées sur l'entreprise et ses produits depuis une page individuelle. Pour extraire plusieurs startups depuis une liste, utilisez plutôt l'agent `startup_listing`.

**Réponse :** Rapport structuré (`ProductResearchReport`) contenant :
- `company` : Profil d'entreprise complet avec :
  - `name` : Nom officiel de l'entreprise
  - `logo_url` : URL absolue du logo si disponible
  - `description` : Description courte de l'entreprise
  - `official_website` : URL du site web principal
  - `linkedin_page` : URL de la page LinkedIn si disponible
  - `other_facts` : Liste de faits notables (financement, métriques, fondateurs, etc.)
- `products` : Liste des produits principaux (min: 1) avec :
  - `product_name` : Nom du produit
  - `what_it_does` : Proposition de valeur en une phrase
  - `go_to_market` : Modèle économique (B2B, B2C, B2G, etc.)
  - `target_audience` : Personas ou industries ciblées
  - `description` : Description plus détaillée du produit

### 2. Startup Listing (`startup_listing`)

Agent de listing de startups qui extrait une liste légère de startups depuis des annuaires (Product Hunt, BetaList, FutureTools, etc.). Idéal pour créer rapidement une liste de startups avec leurs informations de base.

**Endpoint :** `POST /agents/startup_listing/run`

**Paramètres :**
- `url` (requis, `AnyHttpUrl`) : URL de l'annuaire ou de la page de listing à analyser
- `max_startups` (optionnel, `int`, défaut: `12`) : Nombre maximum de startups à capturer (min: 1, max: 1000). Utilisez un nombre élevé comme 1000 pour extraire toutes les startups disponibles
- `output_path` (optionnel, `Path`, défaut: `"startup_listings.json"`) : Chemin de destination pour le fichier JSON

**Exemple complet avec tous les paramètres :**

```bash
curl -X POST "http://localhost:8000/agents/startup_listing/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.producthunt.com/topics/startups",
       "max_startups": 100,
       "output_path": "product_hunt_startups.json"
     }'
```

**Exemple pour BetaList :**

```bash
curl -X POST "http://localhost:8000/agents/startup_listing/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://betalist.com/startups",
       "max_startups": 1000,
       "output_path": "all_betalist_startups.json"
     }'
```

**Note importante :** Pour les plateformes suivantes, préférez les agents spécialisés dédiés (plus performants et fiables) :
- **FutureTools** → `futuretools_extractor` (section 5)
- **AppSumo** → `appsumo_hot_extractor` ou `appsumo_new_extractor` (section 6-7)
- **BetaList** → `betalist_extractor` (section 8)
- **Station F** → `stationf_companies_extractor` (section 9)
- **Zone Secure** → `zone_secure_startups_extractor` (section 10)

L'agent `startup_listing` fonctionne avec ces plateformes mais sera généralement plus lent et moins fiable que les agents spécialisés.

**Exemple pour extraire toutes les startups (limite maximale) :**

```bash
curl -X POST "http://localhost:8000/agents/startup_listing/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.producthunt.com/topics/ai",
       "max_startups": 1000,
       "output_path": "all_ai_startups.json"
     }'
```

**Exemple minimal (seulement l'URL requise) :**

```bash
curl -X POST "http://localhost:8000/agents/startup_listing/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.producthunt.com/topics/ai"
     }'
```

**Exemple avec Python :**

```python
import requests

# Extraire 50 startups depuis Product Hunt avec analyse
response = requests.post(
    "http://localhost:8000/agents/startup_listing/run",
    json={
        "url": "https://www.producthunt.com/topics/startups",
        "max_startups": 50,
        "output_path": "startups.json"
    }
)

# Vérifier le code de statut
if response.status_code == 200:
    # Succès complet
    report = response.json()
    print(f"✅ Extraction réussie!")
    print(f"Source: {report['source_url']}")
    print(f"Startups extraites: {len(report['startups'])}")
elif response.status_code == 206:
    # Rapport de fallback (agent interrompu)
    data = response.json()
    print(f"⚠️  Attention: {data.get('warning', 'Extraction partielle')}")
    print(f"Message: {data.get('message', '')}")
    report = data.get('report', {})
    print(f"Source: {report.get('source_url', 'N/A')}")
    print(f"Startups extraites: {len(report.get('startups', []))}")
    # Le rapport peut contenir des données partielles ou un placeholder
else:
    # Erreur
    print(f"❌ Erreur {response.status_code}: {response.text}")

# Afficher les premières startups (si disponibles)
if 'startups' in report:
    for i, startup in enumerate(report['startups'][:10], 1):
        print(f"\n{i}. {startup['name']}")
        print(f"   Description: {startup.get('description', 'N/A')}")
        print(f"   URL: {startup.get('url', 'N/A')}")
        if startup.get('tags'):
            print(f"   Tags: {', '.join(startup['tags'][:3])}")

# Exemple pour extraire toutes les startups depuis BetaList
response = requests.post(
    "http://localhost:8000/agents/startup_listing/run",
    json={
        "url": "https://betalist.com/startups",
        "max_startups": 1000
    }
)
# Vérifier toujours le code de statut
if response.status_code not in [200, 206]:
    print(f"Erreur: {response.status_code}")
```

**Note importante :** Cet agent extrait une liste de startups depuis une page de listing ou un annuaire. Pour analyser en détail une startup spécifique (entreprise + produits), utilisez plutôt l'agent `product_research` avec l'URL de la page individuelle.

**Réponse :** Rapport structuré (`StartupListingReport`) contenant :
- `source_url` : URL de la page analysée
- `startups` : Liste de profils de startups avec :
  - `name` : Nom de la startup
  - `description` : Description courte
  - `url` : URL de la page de la startup
  - `tags` : Liste des tags/catégories
  - `logo_url` : URL du logo si disponible
  - `website` : Site web officiel si disponible
  - Autres métadonnées selon la source

### 3. Universal Startup Extractor (`universal_startup_extractor`)

**Agent le plus polyvalent** - Extracteur universel de startups qui peut extraire TOUTES les startups depuis N'IMPORTE QUEL site web, quelle que soit sa structure. Utilise des stratégies guidées par LLM pour trouver et extraire les startups de manière exhaustive.

> 💡 **Quand l'utiliser ?** Cet agent est idéal pour des sites web personnalisés ou des annuaires non-standard. Pour les plateformes connues (Product Hunt, AppSumo, BetaList, etc.), préférez les agents spécialisés listés ci-dessous pour de meilleures performances.

**Endpoint :** `POST /agents/universal_startup_extractor/run`

**Paramètres :**
- `url` (requis, `str`) : URL du site web à analyser (peut être n'importe quel site contenant des startups)
- `max_startup` (optionnel, `int`, défaut: `100000`) : Nombre maximum de startups à extraire avant arrêt immédiat (min: 1, max: 1000000). Utilisez un nombre élevé pour extraire toutes les startups disponibles
- `output_path` (optionnel, `Path`, défaut: `"extracted_startups.json"`) : Chemin de destination pour le fichier JSON

**Exemple complet avec tous les paramètres :**

```bash
curl -X POST "http://localhost:8000/agents/universal_startup_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://example.com/startups-directory",
       "max_startup": 500,
       "output_path": "extracted_startups.json"
     }'
```

**Exemple pour un site personnalisé :**

```bash
curl -X POST "http://localhost:8000/agents/universal_startup_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://my-custom-startup-directory.com/companies",
       "max_startup": 1000,
       "output_path": "custom_directory_startups.json"
     }'
```

**Exemple pour extraire toutes les startups (limite très élevée) :**

```bash
curl -X POST "http://localhost:8000/agents/universal_startup_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://example.com/startups",
       "max_startup": 100000,
       "output_path": "all_startups.json"
     }'
```

**Exemple avec limite raisonnable pour un test rapide :**

```bash
curl -X POST "http://localhost:8000/agents/universal_startup_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://example.com/startups",
       "max_startup": 50,
       "output_path": "test_extraction.json"
     }'
```

**Exemple minimal (seulement l'URL requise) :**

```bash
curl -X POST "http://localhost:8000/agents/universal_startup_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://example.com/startups"
     }'
```

**Exemple avec Python :**

```python
import requests

# Extraction avec limite personnalisée et analyse des résultats
response = requests.post(
    "http://localhost:8000/agents/universal_startup_extractor/run",
    json={
        "url": "https://example.com/startups-directory",
        "max_startup": 200,
        "output_path": "my_startups.json"
    }
)
report = response.json()
print(f"Source: {report['source_url']}")
print(f"Startups extraites: {len(report['startups'])}")

# Afficher les premières startups extraites
for i, startup in enumerate(report['startups'][:10], 1):
    print(f"\n{i}. {startup.get('name', 'N/A')}")
    if startup.get('description'):
        print(f"   Description: {startup['description'][:100]}...")
    if startup.get('url'):
        print(f"   URL: {startup['url']}")

# Exemple pour extraction exhaustive
response = requests.post(
    "http://localhost:8000/agents/universal_startup_extractor/run",
    json={
        "url": "https://example.com/startups",
        "max_startup": 100000  # Limite très élevée pour tout extraire
    }
)
```

**Note importante :** Cet agent est conçu pour fonctionner avec n'importe quel site web, même ceux qui ne sont pas spécialement conçus comme annuaires de startups. Il utilise l'IA pour identifier et extraire les startups de manière intelligente. L'extraction peut prendre plus de temps selon la complexité du site et le nombre de startups à extraire. Pour les sites connus (Product Hunt, BetaList, etc.), l'agent `startup_listing` peut être plus rapide et efficace.

**Réponse :** Rapport d'extraction (`StartupExtractionReport`) contenant :
- `source_url` : URL du site analysé
- `startups` : Liste exhaustive de toutes les startups trouvées avec leurs informations complètes :
  - `name` : Nom de la startup
  - `description` : Description détaillée
  - `url` : URL de la page de la startup
  - `website` : Site web officiel si disponible
  - `tags` : Catégories/tags associés
  - Autres métadonnées extraites selon le site

---

## Agents spécialisés par plateforme

### Comment utiliser les agents spécialisés

Les agents spécialisés sont des extracteurs optimisés pour des plateformes spécifiques. Ils offrent plusieurs avantages par rapport aux agents généraux :

#### Avantages des agents spécialisés

1. **Performance optimisée** 🚀
   - Extraction 2-3x plus rapide grâce à une connaissance approfondie de la structure HTML/CSS
   - Moins de tentatives d'extraction, moins de temps de traitement
   - Utilisation efficace de la mémoire et des ressources

2. **Fiabilité accrue** ✅
   - Gestion native des fonctionnalités spécifiques (pagination, scroll infini, authentification)
   - Moins d'erreurs de parsing grâce à des sélecteurs CSS précis
   - Adaptation automatique aux changements mineurs de structure

3. **Données plus complètes** 📊
   - Extraction de métadonnées spécifiques à chaque plateforme (votes, notes, badges, etc.)
   - Meilleure normalisation des données (formats de dates, catégories, etc.)
   - Détection automatique des champs optionnels

4. **Fonctionnalités avancées** 🔧
   - Gestion automatique de la pagination multi-pages
   - Support de l'authentification quand nécessaire
   - Optimisation du scroll infini
   - Interception réseau pour les APIs cachées (Airtable)

#### Structure d'une requête typique

Tous les agents spécialisés suivent le même pattern de requête :

```bash
curl -X POST "http://localhost:8000/agents/{agent_name}/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "...",           # URL de la page (souvent optionnel avec valeur par défaut)
       "max_items": 100,       # Limite d'extraction (nom varie selon l'agent)
       "output_path": "..."    # Chemin de sauvegarde (optionnel)
     }'
```

#### Paramètres communs

La plupart des agents spécialisés partagent des paramètres similaires :

- **`url`** : URL de la page à extraire
  - Souvent optionnel avec une URL par défaut optimale
  - Peut être omis si vous voulez utiliser la page par défaut
  
- **`max_*`** : Limite d'extraction (nom varie : `max_products`, `max_tools`, `max_startups`, etc.)
  - Utilisez une valeur élevée (1000+) pour extraire tout le contenu
  - Par défaut, chaque agent a une limite raisonnable (200-1000)
  
- **`output_path`** : Chemin de sauvegarde du fichier JSON
  - Optionnel, un nom par défaut est fourni
  - Le fichier est sauvegardé dans le répertoire de travail du serveur

#### Bonnes pratiques

1. **Utiliser les valeurs par défaut** : Les agents spécialisés ont des URLs et limites optimisées par défaut. Vous pouvez souvent omettre ces paramètres :
   ```bash
   # Minimal - utilise les valeurs optimales par défaut
   curl -X POST "http://localhost:8000/agents/appsumo_hot_extractor/run" \
        -H "Content-Type: application/json" \
        -d '{}'
   ```

2. **Vérifier le code de statut HTTP** : Les réponses peuvent être :
   - `200` : Succès complet
   - `206` : Extraction partielle (timeout ou interruption)
   - `400` : Paramètres invalides
   - `500` : Erreur serveur

3. **Gérer les timeouts** : Pour de grandes extractions, l'agent peut prendre plusieurs minutes. Soyez patient ou utilisez des limites plus petites pour tester.

4. **Authentification** : Certains agents (comme Station F) supportent l'authentification via les paramètres `email` et `password` si la page est privée.

5. **URLs spécifiques** : Utilisez les URLs recommandées pour chaque plateforme (ex: `/newly-added` pour FutureTools) pour de meilleurs résultats.

#### Exemple complet avec gestion d'erreurs

```python
import requests
import json

def extract_with_retry(agent_name, payload, max_retries=3):
    """Extrait des données avec gestion des erreurs et retry."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"http://localhost:8000/agents/{agent_name}/run",
                json=payload,
                timeout=600  # 10 minutes pour les grandes extractions
            )
            
            if response.status_code == 200:
                return response.json(), "success"
            elif response.status_code == 206:
                data = response.json()
                return data.get('report', {}), "partial"
            elif response.status_code == 400:
                error_msg = response.json().get('detail', 'Invalid request')
                raise ValueError(f"Erreur de validation: {error_msg}")
            else:
                response.raise_for_status()
                
        except requests.Timeout:
            if attempt < max_retries - 1:
                print(f"Timeout, retry {attempt + 1}/{max_retries}...")
                continue
            raise
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Erreur réseau, retry {attempt + 1}/{max_retries}...")
                continue
            raise
    
    raise Exception("Tous les essais ont échoué")

# Utilisation
try:
    report, status = extract_with_retry(
        "futuretools_extractor",
        {
            "url": "https://www.futuretools.io/newly-added",
            "max_tools": 1000
        }
    )
    
    if status == "success":
        print(f"✅ Extraction complète: {len(report.get('tools', []))} outils")
    elif status == "partial":
        print(f"⚠️  Extraction partielle: {len(report.get('tools', []))} outils")
        
    # Traiter les données...
    for tool in report.get('tools', [])[:10]:
        print(f"- {tool.get('name')}: {tool.get('category')}")
        
except Exception as e:
    print(f"❌ Erreur: {e}")
```

#### Différences avec les agents généraux

| Aspect | Agents spécialisés | Agents généraux |
|--------|-------------------|-----------------|
| **Vitesse** | 2-3x plus rapide | Plus lent (exploration) |
| **Fiabilité** | 95%+ de succès | Variable selon le site |
| **Métadonnées** | Spécifiques à la plateforme | Génériques |
| **Configuration** | Valeurs par défaut optimales | Nécessite plus de paramètres |
| **Flexibilité** | Limitée à une plateforme | Fonctionne partout |

#### Quand utiliser un agent spécialisé vs un agent général

**Utilisez un agent spécialisé si :**
- ✅ La plateforme est supportée (voir le guide de sélection)
- ✅ Vous voulez la meilleure performance
- ✅ Vous avez besoin de métadonnées spécifiques
- ✅ Vous extrayez régulièrement de cette plateforme

**Utilisez un agent général si :**
- ✅ La plateforme n'est pas supportée
- ✅ Vous testez un nouveau site
- ✅ Vous avez besoin de flexibilité maximale
- ✅ La structure du site est simple

---

### Liste détaillée des agents spécialisés

### 4. Product Hunt Leaderboard (`product_hunt_leaderboard`)

Agent spécialisé dans l'extraction du leaderboard Product Hunt avec métriques, votes, commentaires, etc. Extrait les produits classés pour une date spécifique.

**Endpoint :** `POST /agents/product_hunt_leaderboard/run`

**Paramètres :**
- `date` (requis, `str`) : Date du leaderboard au format `YYYY-MM-DD` ou `YYYY/MM/DD` (ex: `"2025-01-15"` ou `"2025/01/15"`). La date sera automatiquement normalisée en `YYYY-MM-DD`
- `max_products` (optionnel, `int`, défaut: `1000`) : Nombre maximum de produits à capturer depuis le leaderboard (min: 1, max: 10000). Utilisez un nombre élevé comme 1000 pour extraire tous les produits
- `output_path` (optionnel, `Path`, défaut: `"product_hunt_leaderboard.json"`) : Chemin de destination pour le fichier JSON

**Exemple complet avec tous les paramètres :**

```bash
curl -X POST "http://localhost:8000/agents/product_hunt_leaderboard/run" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2025-01-15",
       "max_products": 1000,
       "output_path": "ph_leaderboard_2025-01-15.json"
     }'
```

**Exemple avec format de date alternatif (YYYY/MM/DD) :**

```bash
curl -X POST "http://localhost:8000/agents/product_hunt_leaderboard/run" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2025/01/15",
       "max_products": 500
     }'
```

**Exemple pour le leaderboard d'aujourd'hui :**

```bash
# Remplacez la date par la date d'aujourd'hui
curl -X POST "http://localhost:8000/agents/product_hunt_leaderboard/run" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2025-01-20",
       "max_products": 1000
     }'
```

**Exemple minimal (seulement la date requise) :**

```bash
curl -X POST "http://localhost:8000/agents/product_hunt_leaderboard/run" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2025-01-15"
     }'
```

**Exemple avec Python :**

```python
import requests
from datetime import datetime, timedelta

# Leaderboard d'hier
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
response = requests.post(
    "http://localhost:8000/agents/product_hunt_leaderboard/run",
    json={
        "date": yesterday,
        "max_products": 1000,
        "output_path": f"ph_leaderboard_{yesterday}.json"
    }
)
leaderboard = response.json()
print(f"Produits extraits: {len(leaderboard['products'])}")
for product in leaderboard['products'][:5]:
    print(f"#{product.get('rank', 'N/A')} - {product['name']}: {product.get('upvotes', 0)} upvotes")
```

**Note importante :** La date est utilisée pour construire automatiquement l'URL du leaderboard Product Hunt au format `https://www.producthunt.com/leaderboard/daily/YYYY/MM/DD/all`. Vous n'avez pas besoin de fournir l'URL complète, seulement la date.

**Réponse :** Rapport structuré (`ProductHuntLeaderboardReport`) contenant :
- `source_url` : URL du leaderboard analysé
- `products` : Liste des produits classés avec leurs métriques (nom, rang, description, tags, upvotes, maker, commentaires, etc.)

### 5. FutureTools Extractor (`futuretools_extractor`)

Agent spécialisé dans l'extraction d'outils depuis FutureTools. Optimisé pour la structure spécifique de FutureTools et utilise des stratégies d'extraction directe du HTML pour une meilleure performance.

**Endpoint :** `POST /agents/futuretools_extractor/run`

**Paramètres :**
- `url` (optionnel, `AnyHttpUrl`, défaut: `"https://www.futuretools.io/newly-added"`) : URL de la page FutureTools à analyser. La page `newly-added` est recommandée car elle contient tous les outils récemment ajoutés
- `max_tools` (optionnel, `int`, défaut: `1000`) : Nombre maximum d'outils à capturer (min: 1, max: 10000). Utilisez un nombre élevé comme 1000 pour extraire tous les outils disponibles
- `output_path` (optionnel, `Path`, défaut: `"futuretools_tools.json"`) : Chemin de destination pour le fichier JSON

**Exemple complet avec tous les paramètres :**

```bash
curl -X POST "http://localhost:8000/agents/futuretools_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.futuretools.io/newly-added",
       "max_tools": 1000,
       "output_path": "futuretools_tools.json"
     }'
```

**Exemple pour la page newly-added (recommandé) :**

```bash
curl -X POST "http://localhost:8000/agents/futuretools_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.futuretools.io/newly-added",
       "max_tools": 1000
     }'
```

**Exemple pour la page principale :**

```bash
curl -X POST "http://localhost:8000/agents/futuretools_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.futuretools.io/",
       "max_tools": 500
     }'
```

**Exemple minimal (utilise les valeurs par défaut) :**

```bash
curl -X POST "http://localhost:8000/agents/futuretools_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.futuretools.io/newly-added"
     }'
```

**Exemple avec Python :**

```python
import requests

# Extraction depuis la page newly-added (recommandé)
response = requests.post(
    "http://localhost:8000/agents/futuretools_extractor/run",
    json={
        "url": "https://www.futuretools.io/newly-added",
        "max_tools": 1000,
        "output_path": "futuretools_tools.json"
    }
)

# Vérifier le code de statut
if response.status_code == 200:
    report = response.json()
    print(f"✅ Extraction réussie!")
    print(f"Source: {report['source_url']}")
    print(f"Outils extraits: {len(report['tools'])}")
    
    # Afficher les premiers outils
    for i, tool in enumerate(report['tools'][:10], 1):
        print(f"\n{i}. {tool['name']}")
        print(f"   Catégorie: {tool.get('category', 'N/A')}")
        print(f"   Description: {tool.get('description', 'N/A')[:100]}...")
        if tool.get('tool_url'):
            print(f"   URL: {tool['tool_url']}")
elif response.status_code == 206:
    data = response.json()
    print(f"⚠️  Attention: {data.get('warning', 'Extraction partielle')}")
else:
    print(f"❌ Erreur {response.status_code}: {response.text}")
```

**Note importante :** Cet agent est spécialement optimisé pour FutureTools et utilise des stratégies d'extraction directe du HTML pour une meilleure performance. Pour d'autres sites, utilisez `startup_listing` ou `universal_startup_extractor`. La page `newly-added` est recommandée car elle contient tous les outils récemment ajoutés dans une structure cohérente.

**Réponse :** Rapport structuré (`FutureToolsReport`) contenant :
- `source_url` : URL de la page FutureTools analysée
- `tools` : Liste des outils extraits avec :
  - `name` : Nom de l'outil
  - `tool_url` : URL de la page de l'outil si disponible
  - `category` : Catégorie/tag de l'outil (ex: "Automation & Agents", "Productivity")
  - `description` : Description de l'outil si disponible

### 6. AppSumo Hot Extractor (`appsumo_hot_extractor`)

Agent spécialisé dans l'extraction de produits tendances depuis la collection "What's hot" d'AppSumo. Extrait les produits avec leurs prix, notes, badges et informations de catégorie.

**Endpoint :** `POST /agents/appsumo_hot_extractor/run`

**Paramètres :**
- `url` (optionnel, `AnyHttpUrl`, défaut: `"https://appsumo.com/collections/whats-hot/"`) : URL de la collection "What's hot" d'AppSumo
- `max_products` (optionnel, `int`, défaut: `200`) : Nombre maximum de produits à capturer (min: 1, max: 2000)
- `output_path` (optionnel, `Path`, défaut: `"appsumo_hot_products.json"`) : Chemin de destination pour le fichier JSON

**Exemple complet :**

```bash
curl -X POST "http://localhost:8000/agents/appsumo_hot_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://appsumo.com/collections/whats-hot/",
       "max_products": 200,
       "output_path": "appsumo_hot_products.json"
     }'
```

**Exemple minimal :**

```bash
curl -X POST "http://localhost:8000/agents/appsumo_hot_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{}'
```

**Exemple avec Python :**

```python
import requests

response = requests.post(
    "http://localhost:8000/agents/appsumo_hot_extractor/run",
    json={
        "url": "https://appsumo.com/collections/whats-hot/",
        "max_products": 500
    }
)
report = response.json()
print(f"Produits extraits: {len(report['products'])}")
```

**Réponse :** Rapport structuré (`AppSumoHotReport`) contenant les produits tendances avec leurs métadonnées complètes.

### 7. AppSumo New Extractor (`appsumo_new_extractor`)

Agent spécialisé dans l'extraction de nouveaux produits depuis la collection "New arrivals" d'AppSumo. Extrait les produits récemment ajoutés avec leurs prix, notes, badges et informations de catégorie.

**Endpoint :** `POST /agents/appsumo_new_extractor/run`

**Paramètres :**
- `url` (optionnel, `AnyHttpUrl`, défaut: `"https://appsumo.com/collections/new/"`) : URL de la collection "New arrivals" d'AppSumo
- `max_products` (optionnel, `int`, défaut: `200`) : Nombre maximum de produits à capturer (min: 1, max: 2000)
- `output_path` (optionnel, `Path`, défaut: `"appsumo_new_products.json"`) : Chemin de destination pour le fichier JSON

**Exemple complet :**

```bash
curl -X POST "http://localhost:8000/agents/appsumo_new_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://appsumo.com/collections/new/",
       "max_products": 200,
       "output_path": "appsumo_new_products.json"
     }'
```

**Exemple minimal :**

```bash
curl -X POST "http://localhost:8000/agents/appsumo_new_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{}'
```

### 8. BetaList Extractor (`betalist_extractor`)

Agent spécialisé dans l'extraction de startups depuis BetaList. Extrait les startups récemment publiées avec filtrage par date. Optimisé pour le scroll infini de BetaList.

**Endpoint :** `POST /agents/betalist_extractor/run`

**Paramètres :**
- `url` (optionnel, `AnyHttpUrl`, défaut: `"https://betalist.com/"`) : URL de la page BetaList
- `last_days` (optionnel, `int`, défaut: `3`) : Nombre de jours récents à conserver (min: 1, max: 30)
- `max_startups` (optionnel, `int`, défaut: `200`) : Nombre maximum de startups à capturer (min: 1, max: 2000)
- `output_path` (optionnel, `Path`, défaut: `"betalist_recent.json"`) : Chemin de destination pour le fichier JSON

**Exemple complet :**

```bash
curl -X POST "http://localhost:8000/agents/betalist_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://betalist.com/",
       "last_days": 7,
       "max_startups": 500,
       "output_path": "betalist_last_week.json"
     }'
```

**Exemple pour les startups de la dernière semaine :**

```bash
curl -X POST "http://localhost:8000/agents/betalist_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "last_days": 7,
       "max_startups": 1000
     }'
```

### 9. Station F Companies Extractor (`stationf_companies_extractor`)

Agent spécialisé dans l'extraction d'entreprises depuis Station F HAL (Hub & Accelerator). Extrait les entreprises avec leurs secteurs, stades, localisations et autres métadonnées. Supporte l'authentification optionnelle pour accéder aux pages privées.

**Endpoint :** `POST /agents/stationf_companies_extractor/run`

**Paramètres :**
- `url` (optionnel, `str`, défaut: `"https://hal2.stationf.co/companies"`) : URL de la page des entreprises Station F
- `max_companies` (optionnel, `int`, défaut: `1000`) : Nombre maximum d'entreprises à capturer (min: 1, max: 10000)
- `output_path` (optionnel, `Path`, défaut: `"stationf_companies.json"`) : Chemin de destination pour le fichier JSON
- `email` (optionnel, `str`) : Email pour l'authentification (si la page nécessite une connexion)
- `password` (optionnel, `str`) : Mot de passe pour l'authentification (si la page nécessite une connexion)

**Exemple sans authentification :**

```bash
curl -X POST "http://localhost:8000/agents/stationf_companies_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://hal2.stationf.co/companies",
       "max_companies": 1000,
       "output_path": "stationf_companies.json"
     }'
```

**Exemple avec authentification :**

```bash
curl -X POST "http://localhost:8000/agents/stationf_companies_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://hal2.stationf.co/companies",
       "max_companies": 1000,
       "email": "votre@email.com",
       "password": "votre_mot_de_passe",
       "output_path": "stationf_companies.json"
     }'
```

**Note importante :** Si la page nécessite une authentification, fournissez les paramètres `email` et `password`. Sinon, l'agent tentera d'accéder à la page publique si disponible.

### 10. Zone Secure Startups Extractor (`zone_secure_startups_extractor`)

Agent spécialisé dans l'extraction EXHAUSTIVE de startups depuis Zone Secure. Extrait toutes les startups de toutes les pages avec navigation multi-pages. Gère la pagination et le filtrage des éléments de navigation.

**Endpoint :** `POST /agents/zone_secure_startups_extractor/run`

**Paramètres :**
- `url` (optionnel, `str`, défaut: `"https://fr.zone-secure.net/20412/2540033/#page=1"`) : URL de la première page des startups Zone Secure
- `max_startups` (optionnel, `int`, défaut: `10000`) : Nombre maximum de startups à capturer (min: 1, max: 50000)
- `output_path` (optionnel, `Path`, défaut: `"zone_secure_startups.json"`) : Chemin de destination pour le fichier JSON

**Exemple complet :**

```bash
curl -X POST "http://localhost:8000/agents/zone_secure_startups_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://fr.zone-secure.net/20412/2540033/#page=1",
       "max_startups": 10000,
       "output_path": "zone_secure_startups.json"
     }'
```

**Exemple pour extraction limitée :**

```bash
curl -X POST "http://localhost:8000/agents/zone_secure_startups_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "max_startups": 100
     }'
```

**Note importante :** Cet agent est conçu pour extraire toutes les startups de toutes les pages. Il gère automatiquement la pagination et navigue entre les pages jusqu'à atteindre la limite `max_startups` ou la fin du catalogue.

### 11. Airtable Extractor (`airtable_extractor`)

Agent spécialisé dans l'extraction de données depuis Airtable. Extrait les lignes et colonnes depuis une vue partagée Airtable. Utilise l'interception réseau pour récupérer l'URL API automatiquement.

**Endpoint :** `POST /agents/airtable_extractor/run`

**Paramètres :**
- `url` (requis, `str`) : URL de la vue partagée Airtable (format: `https://airtable.com/appXXX/shrXXX`) ou URL API complète

**Exemple avec vue partagée :**

```bash
curl -X POST "http://localhost:8000/agents/airtable_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://airtable.com/appXXXXXXXXXXXXXX/tblYYYYYYYYYYYYYY/viwZZZZZZZZZZZZZZ"
     }'
```

**Exemple avec endpoint API :**

```bash
curl -X POST "http://localhost:8000/agents/airtable_extractor/run" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://api.airtable.com/v0/appXXXXXXXXXXXXXX/tblYYYYYYYYYYYYYY"
     }'
```

**Exemple avec Python :**

```python
import requests

response = requests.post(
    "http://localhost:8000/agents/airtable_extractor/run",
    json={
        "url": "https://airtable.com/appXXX/tblYYY/viwZZZ"
    }
)
data = response.json()
print(f"Colonnes: {len(data['columns'])}")
print(f"Lignes: {len(data['rows'])}")
```

**Réponse :** Rapport structuré (`AirtableReport`) contenant :
- `metadata` : Métadonnées sur la source et les statistiques
- `columns` : Liste des colonnes avec leurs identifiants et noms
- `rows` : Liste des lignes avec toutes les données

**Note importante :** Si vous fournissez une vue partagée Airtable (URL publique), l'agent intercepte automatiquement les requêtes réseau pour récupérer l'URL API et extraire les données. Pour les endpoints API directs, l'agent les utilise directement.

### Résumé des agents spécialisés

Tous les agents spécialisés offrent :
- ✅ **Performance optimale** : Extraction 2-3x plus rapide que les agents généraux
- ✅ **Fiabilité accrue** : Moins d'erreurs grâce à une connaissance approfondie de la structure
- ✅ **Données complètes** : Métadonnées spécifiques à chaque plateforme
- ✅ **Gestion avancée** : Pagination, authentification, scroll infini automatiques

Pour toute autre plateforme non listée ci-dessus, utilisez l'agent `universal_startup_extractor` qui fonctionne avec n'importe quel site web.

---

## Endpoints généraux

### Lister les agents

```bash
GET /agents
```

Retourne la liste de tous les agents disponibles avec leurs descriptions.

### Obtenir les métadonnées d'un agent

```bash
GET /agents/{agent_name}
```

Retourne les informations détaillées d'un agent spécifique (description, schémas d'entrée/sortie).

### Healthcheck

```bash
GET /health
```

Vérifie que l'API est opérationnelle et retourne le nombre d'agents disponibles.

## Documentation interactive

L'API expose une documentation interactive complète avec des exemples :

- **Swagger UI** : `http://localhost:8000/docs` - Interface interactive pour tester tous les endpoints
- **ReDoc** : `http://localhost:8000/redoc` - Documentation alternative avec une présentation élégante

Chaque endpoint POST contient plusieurs exemples pré-configurés que vous pouvez tester directement depuis l'interface Swagger.

## Structure des réponses

Tous les endpoints POST retournent des rapports structurés au format JSON avec des schémas Pydantic validés. Les erreurs sont retournées avec des codes HTTP appropriés :

- `200` : Succès - Rapport généré avec succès
- `206` : Contenu partiel - L'agent a été interrompu avant de finaliser l'extraction. La réponse contient un rapport de fallback avec un champ `warning` expliquant le problème. Vérifiez les logs du serveur pour plus de détails.
- `400` : Requête invalide (payload mal formé, paramètres invalides)
- `404` : Agent introuvable
- `500` : Erreur interne lors de l'exécution de l'agent

### Gestion des rapports de fallback

Si un agent est interrompu avant de finaliser l'extraction (timeout, erreur, etc.), l'API retourne un code `206` (Partial Content) avec une structure de réponse enrichie :

```json
{
  "report": {
    "source_url": "...",
    "startups": [{"name": "Informations indisponibles", ...}]
  },
  "warning": "L'agent a été interrompu avant de finaliser l'extraction...",
  "success": false,
  "message": "Vérifiez les logs du serveur pour plus de détails..."
}
```

**Causes possibles d'un rapport de fallback :**
- Timeout de l'agent (page trop lente à charger, trop de contenu)
- Erreur de parsing JSON par le LLM
- Problème de connexion ou de chargement de la page
- Configuration LLM incorrecte (clé API manquante ou invalide)

**Solutions :**
1. Vérifiez les logs du serveur pour identifier la cause exacte
2. Vérifiez que les variables d'environnement LLM sont correctement configurées (`BROWSER_USE_API_KEY` ou `OPENAI_API_KEY`)
3. Réessayez avec une URL plus simple ou une limite plus faible
4. Augmentez les timeouts si nécessaire (configuration dans le code de l'agent)

## Personnalisation

Pour ajouter un nouvel agent à l'API :

1. Importez les classes Input, Report et la fonction `run_*` de votre agent
2. Enregistrez l'agent dans le registre :

```python
from examples.histia.fastapi_agents import registry
from examples.histia.votre_agent import (
    VotreAgentInput,
    VotreAgentReport,
    run_votre_agent,
)

registry.register(
    name='votre_agent',
    description='Description de votre agent',
    input_class=VotreAgentInput,
    output_class=VotreAgentReport,
    run_function=run_votre_agent,
)
```

L'endpoint sera automatiquement créé à `/agents/votre_agent/run`.

## Notes importantes

- Les agents utilisent `browser-use` pour l'automation web et peuvent prendre plusieurs minutes selon la complexité de la tâche
- Les timeouts sont configurés pour chaque agent selon ses besoins (généralement 300s pour step_timeout, 180s pour llm_timeout)
- Les rapports sont validés avec Pydantic pour garantir la cohérence des données
- En cas d'échec partiel, les agents retournent des rapports de fallback (code 206) plutôt que d'échouer complètement
- **Important** : Vérifiez toujours le code de statut HTTP dans vos clients :
  - `200` = Succès complet
  - `206` = Rapport de fallback (agent interrompu, vérifiez les logs)
  - `400` = Requête invalide
  - `500` = Erreur serveur

### Dépannage des erreurs d'extraction

Si vous recevez des rapports de fallback (code 206) :

1. **Vérifiez les variables d'environnement** :
   ```bash
   # Pour ChatBrowserUse (recommandé)
   export BROWSER_USE_API_KEY="votre_clé"
   
   # Ou pour OpenAI/LiteLLM
   export OPENAI_API_KEY="votre_clé"
   export OPENAI_API_BASE="https://votre-endpoint.com"  # Si nécessaire
   ```

2. **Vérifiez les logs du serveur** : Les agents affichent des messages détaillés sur la console où l'API est lancée

3. **Réduisez la complexité** :
   - Utilisez des limites plus faibles (`max_startups`, `max_products`)
   - Testez avec des URLs plus simples d'abord
   - Vérifiez que l'URL est accessible et contient bien du contenu

4. **Vérifiez la connectivité** : Assurez-vous que le serveur peut accéder aux URLs cibles (pas de firewall, proxy, etc.)
