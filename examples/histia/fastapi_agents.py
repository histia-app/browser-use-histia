"""API FastAPI pour interroger les agents Histia d'extraction de données.

Ce module expose une API REST complète pour utiliser les agents d'extraction Histia.
Chaque agent dispose de son propre endpoint POST dédié avec documentation complète.

Pour lancer le serveur :
    uvicorn examples.histia.fastapi_agents:app --reload

Variables d'environnement utiles :
- ``OPENAI_API_KEY`` / ``OPENAI_API_BASE`` pour le backend LiteLLM.
- ``BROWSER_USE_API_KEY`` pour utiliser ChatBrowserUse (recommandé).
- ``LLM_BACKEND=gemini`` et ``GOOGLE_API_KEY`` (ou ``GEMINI_API_KEY``) pour basculer sur Gemini.
"""

from functools import lru_cache
from typing import Any, Callable, Dict, List, Type, TypeVar

from fastapi import Body, Depends, FastAPI, HTTPException, Path
from pydantic import BaseModel, field_serializer

# Import des agents et leurs schémas
from examples.histia.product_research_agent import (
    ProductResearchInput,
    ProductResearchReport,
    run_product_research,
)
from examples.histia.startup_listing_agent import (
    StartupListingInput,
    StartupListingReport,
    run_startup_listing,
)
from examples.histia.universal_startup_extractor import (
    StartupExtractionReport,
    UniversalStartupExtractorInput,
    run_universal_startup_extraction,
)
from examples.histia.product_hunt_leaderboard_agent import (
    ProductHuntLeaderboardInput,
    ProductHuntLeaderboardReport,
    run_product_hunt_leaderboard,
)
from examples.histia.futuretools_extractor import (
    FutureToolsInput,
    FutureToolsReport,
    run_futuretools_extraction,
)
from examples.histia.appsumo_hot_extractor import (
    AppSumoHotInput,
    AppSumoHotReport,
    run_appsumo_hot_extraction,
)
from examples.histia.appsumo_new_extractor import (
    AppSumoNewInput,
    AppSumoNewReport,
    run_appsumo_new_extraction,
)
from examples.histia.betalist_extractor import (
    BetalistInput,
    BetalistReport,
    run_betalist_extraction,
)
from examples.histia.stationf_companies_agent import (
    StationFCompaniesInput,
    StationFCompaniesReport,
    run_stationf_companies,
)
from examples.histia.zone_secure_startups_agent import (
    ZoneSecureStartupsInput,
    ZoneSecureStartupsReport,
    run_zone_secure_startups,
)

# Wrapper async pour Airtable
import asyncio
from pathlib import Path as PathLibPath
from pydantic import AnyHttpUrl, Field


class AirtableInput(BaseModel):
    """User-provided parameters for the Airtable extraction task."""

    url: AnyHttpUrl = Field(
        ...,
        description='URL de la vue partagée Airtable (format: https://airtable.com/appXXX/shrXXX ou URL API complète)',
    )
    output_path: PathLibPath = Field(
        default=PathLibPath('airtable_extracted.json'),
        description='Destination pour le fichier JSON des données extraites',
    )


class AirtableReport(BaseModel):
    """Complete response returned by the Airtable extractor."""

    source_url: AnyHttpUrl = Field(..., description='URL Airtable qui a été analysée')
    metadata: Dict[str, Any] = Field(..., description='Métadonnées de la base (nombre de colonnes, de lignes, etc.)')
    columns: List[Dict[str, str]] = Field(..., description='Liste des colonnes avec leur ID et nom')
    rows: List[Dict[str, Any]] = Field(..., description='Lignes de données extraites')

    @field_serializer('source_url')
    def serialize_source_url(self, value: AnyHttpUrl, _info) -> str:
        """Convert AnyHttpUrl to string for JSON serialization."""
        return str(value)


async def run_airtable_extraction(task_input: AirtableInput) -> AirtableReport:
    """Execute the Airtable extraction and return the structured data."""
    from examples.histia.airtable_extract import extract_airtable_data, extract_columns, extract_rows

    # Extract data (synchronous function, run in executor)
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, extract_airtable_data, str(task_input.url))

    # Extract columns
    columns_dict = extract_columns(data)
    columns = [{'id': col_id, 'name': col_name} for col_id, col_name in columns_dict.items()]

    # Extract rows
    rows = extract_rows(data, columns_dict)

    # Build metadata
    metadata = {
        'source': str(task_input.url),
        'total_columns': len(columns),
        'total_rows': len(rows),
    }

    return AirtableReport(
        source_url=task_input.url,
        metadata=metadata,
        columns=columns,
        rows=rows,
    )


TInput = TypeVar('TInput', bound=BaseModel)
TReport = TypeVar('TReport', bound=BaseModel)


class AgentInfo(BaseModel):
    """Description légère d'un agent exposé par l'API."""

    name: str
    description: str
    input_schema: str
    output_schema: str


class AgentRegistry:
    """Registre centralisé des agents disponibles."""

    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        input_class: Type[TInput],
        output_class: Type[TReport],
        run_function: Callable[[TInput], Any],
    ):
        """Enregistre un agent dans le registre."""
        self._agents[name] = {
            'description': description,
            'input_class': input_class,
            'output_class': output_class,
            'run_function': run_function,
        }

    def get_agent_info(self, name: str) -> Dict[str, Any]:
        """Récupère les informations d'un agent."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found")
        return self._agents[name]

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """Liste tous les agents enregistrés."""
        return self._agents.copy()


# Création du registre et enregistrement des agents
registry = AgentRegistry()

registry.register(
    name='product_research',
    description=(
        'Agent de recherche de produits qui extrait des informations structurées '
        'sur les entreprises et leurs produits depuis des listings (Product Hunt, BetaList, etc.). '
        'Retourne un profil d\'entreprise, les produits principaux, et des faits notables.'
    ),
    input_class=ProductResearchInput,
    output_class=ProductResearchReport,
    run_function=run_product_research,
)

registry.register(
    name='startup_listing',
    description=(
        'Agent de listing de startups qui extrait une liste légère de startups '
        'depuis des annuaires (Product Hunt, BetaList, FutureTools, etc.). '
        'Idéal pour créer rapidement une liste de startups avec leurs informations de base.'
    ),
    input_class=StartupListingInput,
    output_class=StartupListingReport,
    run_function=run_startup_listing,
)

registry.register(
    name='universal_startup_extractor',
    description=(
        'Extracteur universel de startups qui peut extraire TOUTES les startups '
        'depuis N\'IMPORTE QUEL site web, quelle que soit sa structure. '
        'Utilise des stratégies guidées par LLM pour trouver et extraire les startups de manière exhaustive.'
    ),
    input_class=UniversalStartupExtractorInput,
    output_class=StartupExtractionReport,
    run_function=run_universal_startup_extraction,
)

registry.register(
    name='product_hunt_leaderboard',
    description=(
        'Agent spécialisé dans l\'extraction du leaderboard Product Hunt. '
        'Extrait les produits classés avec leurs métriques, votes, commentaires, etc.'
    ),
    input_class=ProductHuntLeaderboardInput,
    output_class=ProductHuntLeaderboardReport,
    run_function=run_product_hunt_leaderboard,
)

registry.register(
    name='futuretools_extractor',
    description=(
        'Agent spécialisé dans l\'extraction d\'outils depuis FutureTools. '
        'Optimisé pour la structure spécifique de FutureTools (page newly-added). '
        'Extrait les outils avec leurs catégories, descriptions et URLs.'
    ),
    input_class=FutureToolsInput,
    output_class=FutureToolsReport,
    run_function=run_futuretools_extraction,
)

registry.register(
    name='appsumo_hot_extractor',
    description=(
        'Agent spécialisé dans l\'extraction de produits tendances depuis AppSumo "What\'s hot". '
        'Extrait les produits avec leurs prix, notes, badges et informations de catégorie.'
    ),
    input_class=AppSumoHotInput,
    output_class=AppSumoHotReport,
    run_function=run_appsumo_hot_extraction,
)

registry.register(
    name='appsumo_new_extractor',
    description=(
        'Agent spécialisé dans l\'extraction de nouveaux produits depuis AppSumo "New arrivals". '
        'Extrait les produits récemment ajoutés avec leurs prix, notes, badges et informations de catégorie.'
    ),
    input_class=AppSumoNewInput,
    output_class=AppSumoNewReport,
    run_function=run_appsumo_new_extraction,
)

registry.register(
    name='betalist_extractor',
    description=(
        'Agent spécialisé dans l\'extraction de startups depuis BetaList. '
        'Extrait les startups récemment publiées avec filtrage par date. '
        'Optimisé pour le scroll infini de BetaList.'
    ),
    input_class=BetalistInput,
    output_class=BetalistReport,
    run_function=run_betalist_extraction,
)

registry.register(
    name='stationf_companies_extractor',
    description=(
        'Agent spécialisé dans l\'extraction d\'entreprises depuis Station F HAL. '
        'Extrait les entreprises avec leurs secteurs, stades, localisations et autres métadonnées. '
        'Supporte l\'authentification optionnelle pour accéder aux pages privées.'
    ),
    input_class=StationFCompaniesInput,
    output_class=StationFCompaniesReport,
    run_function=run_stationf_companies,
)

registry.register(
    name='zone_secure_startups_extractor',
    description=(
        'Agent spécialisé dans l\'extraction EXHAUSTIVE de startups depuis Zone Secure. '
        'Extrait toutes les startups de toutes les pages avec navigation multi-pages. '
        'Gère la pagination et le filtrage des éléments de navigation.'
    ),
    input_class=ZoneSecureStartupsInput,
    output_class=ZoneSecureStartupsReport,
    run_function=run_zone_secure_startups,
)

registry.register(
    name='airtable_extractor',
    description=(
        'Agent spécialisé dans l\'extraction de données depuis Airtable. '
        'Extrait les lignes et colonnes depuis une vue partagée Airtable. '
        'Utilise l\'interception réseau pour récupérer l\'URL API automatiquement.'
    ),
    input_class=AirtableInput,
    output_class=AirtableReport,
    run_function=run_airtable_extraction,
)


app = FastAPI(
    title='API Agents Histia',
    description=(
        'API REST complète pour utiliser les agents d\'extraction Histia. '
        'Chaque agent dispose d\'un endpoint POST dédié avec documentation complète et exemples.'
    ),
    version='1.0.0',
)


@app.get(
    '/agents',
    response_model=List[AgentInfo],
    summary='Lister les agents disponibles',
    description=(
        'Retourne la liste de tous les agents d\'extraction configurés dans l\'API. '
        'Chaque agent dispose d\'un endpoint POST dédié pour exécuter des tâches d\'extraction.'
    ),
    responses={
        200: {
            'description': 'Liste des agents disponibles',
        },
    },
)
def list_agents() -> List[AgentInfo]:
    """Retourne la liste des agents configurés.

    **Exemple d'utilisation :**

    ```bash
    # Avec curl
    curl http://localhost:8000/agents

    # Avec Python requests
    import requests
    response = requests.get('http://localhost:8000/agents')
    print(response.json())
    ```
    """
    agents = []
    for name, info in registry.list_agents().items():
        agents.append(
            AgentInfo(
                name=name,
                description=info['description'],
                input_schema=info['input_class'].__name__,
                output_schema=info['output_class'].__name__,
            )
        )
    return agents


@app.get(
    '/agents/{agent_name}',
    response_model=AgentInfo,
    summary='Obtenir les métadonnées d\'un agent',
    description=(
        'Retourne les informations détaillées d\'un agent spécifique, '
        'incluant son nom, sa description et les schémas d\'entrée/sortie.'
    ),
    responses={
        200: {
            'description': 'Informations de l\'agent',
        },
        404: {'description': 'Agent introuvable'},
    },
)
def describe_agent(
    agent_name: str = Path(
        ...,
        description='Nom de l\'agent',
        examples=['product_research', 'startup_listing', 'universal_startup_extractor'],
    )
) -> AgentInfo:
    """Expose les détails d'un agent unique.

    **Exemples d'utilisation :**

    ```bash
    # Avec curl
    curl http://localhost:8000/agents/product_research

    # Avec Python requests
    import requests
    response = requests.get('http://localhost:8000/agents/product_research')
    print(response.json())
    ```
    """
    try:
        info = registry.get_agent_info(agent_name)
        return AgentInfo(
            name=agent_name,
            description=info['description'],
            input_schema=info['input_class'].__name__,
            output_schema=info['output_class'].__name__,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f'Agent introuvable: {agent_name}') from exc


def _create_agent_endpoint(
    agent_name: str,
    input_class: Type[TInput],
    output_class: Type[TReport],
    run_function: Callable[[TInput], Any],
    description: str,
):
    """Crée dynamiquement un endpoint POST pour un agent."""

    @app.post(
        f'/agents/{agent_name}/run',
        response_model=output_class,
        summary=f'Exécuter l\'agent {agent_name}',
        response_description=f'Rapport structuré retourné par l\'agent {agent_name}',
        description=description,
        responses={
            200: {
                'description': 'Rapport d\'extraction généré avec succès',
            },
            400: {'description': 'Requête invalide (payload mal formé)'},
            500: {'description': 'Erreur interne lors de l\'exécution de l\'agent'},
        },
    )
    async def run_agent(
        payload: input_class = Body(  # type: ignore[arg-type,assignment]
            ...,
            examples=_get_examples_for_agent(agent_name, input_class),  # type: ignore[arg-type]
        ),
    ) -> output_class:  # type: ignore[misc]
        """Exécute l'agent avec les paramètres fournis.

        Consultez la documentation de chaque agent pour connaître les paramètres spécifiques :

        **Agents de recherche et extraction :**
        - `product_research` : nécessite `url` (requis), `max_products` (optionnel, défaut: 3), `output_path` (optionnel)
        - `startup_listing` : nécessite `url` (requis), `max_startups` (optionnel, défaut: 12), `output_path` (optionnel)
        - `universal_startup_extractor` : nécessite `url` (requis), `max_startup` (optionnel, défaut: 100000), `output_path` (optionnel)

        **Agents spécialisés par plateforme :**
        - `product_hunt_leaderboard` : nécessite `date` (requis, format YYYY-MM-DD ou YYYY/MM/DD), `max_products` (optionnel, défaut: 1000), `output_path` (optionnel)
        - `futuretools_extractor` : nécessite `url` (optionnel, défaut: https://www.futuretools.io/newly-added), `max_tools` (optionnel, défaut: 1000), `output_path` (optionnel)
        - `airtable_extractor` : nécessite `url` (requis), URL d'une vue partagée Airtable ou endpoint API
        - `appsumo_hot_extractor` : nécessite `url` (optionnel, défaut: https://appsumo.com/collections/whats-hot/), `max_products` (optionnel, défaut: 200), `output_path` (optionnel)
        - `appsumo_new_extractor` : nécessite `url` (optionnel, défaut: https://appsumo.com/collections/new/), `max_products` (optionnel, défaut: 200), `output_path` (optionnel)
        - `betalist_extractor` : nécessite `url` (optionnel, défaut: https://betalist.com/), `last_days` (optionnel, défaut: 3), `max_startups` (optionnel, défaut: 200), `output_path` (optionnel)
        - `stationf_companies_extractor` : nécessite `url` (optionnel, défaut: https://hal2.stationf.co/companies), `max_companies` (optionnel, défaut: 1000), `output_path` (optionnel), `email` (optionnel), `password` (optionnel)
        - `zone_secure_startups_extractor` : nécessite `url` (optionnel, défaut: https://fr.zone-secure.net/20412/2540033/#page=1), `max_startups` (optionnel, défaut: 10000), `output_path` (optionnel)

        **Exemples d'utilisation :**

        ```bash
        # Product Research
        curl -X POST "http://localhost:8000/agents/product_research/run" \\
             -H "Content-Type: application/json" \\
             -d '{"url": "https://www.producthunt.com/posts/example", "max_products": 3}'

        # Product Hunt Leaderboard
        curl -X POST "http://localhost:8000/agents/product_hunt_leaderboard/run" \\
             -H "Content-Type: application/json" \\
             -d '{"date": "2025-01-15", "max_products": 1000}'

        # FutureTools Extractor
        curl -X POST "http://localhost:8000/agents/futuretools_extractor/run" \\
             -H "Content-Type: application/json" \\
             -d '{"url": "https://www.futuretools.io/newly-added", "max_tools": 1000}'

        # Airtable Extractor
        curl -X POST "http://localhost:8000/agents/airtable_extractor/run" \\
             -H "Content-Type: application/json" \\
             -d '{"url": "https://airtable.com/appXXX/tblYYY/viwZZZ"}'

        # AppSumo Hot Extractor
        curl -X POST "http://localhost:8000/agents/appsumo_hot_extractor/run" \\
             -H "Content-Type: application/json" \\
             -d '{"url": "https://appsumo.com/collections/whats-hot/", "max_products": 200}'

        # BetaList Extractor
        curl -X POST "http://localhost:8000/agents/betalist_extractor/run" \\
             -H "Content-Type: application/json" \\
             -d '{"url": "https://betalist.com/", "last_days": 7, "max_startups": 500}'

        # Station F Companies (avec authentification)
        curl -X POST "http://localhost:8000/agents/stationf_companies_extractor/run" \\
             -H "Content-Type: application/json" \\
             -d '{"url": "https://hal2.stationf.co/companies", "email": "user@example.com", "password": "secret"}'
        ```

        ```python
        # Avec Python requests
        import requests

        # Product Research
        response = requests.post(
            "http://localhost:8000/agents/product_research/run",
            json={"url": "https://www.producthunt.com/posts/example", "max_products": 3}
        )

        # FutureTools Extractor
        response = requests.post(
            "http://localhost:8000/agents/futuretools_extractor/run",
            json={"url": "https://www.futuretools.io/newly-added", "max_tools": 1000}
        )

        # Airtable Extractor
        response = requests.post(
            "http://localhost:8000/agents/airtable_extractor/run",
            json={"url": "https://airtable.com/appXXX/tblYYY/viwZZZ"}
        )
        ```
        """
        try:
            result = await run_function(payload)
            if result is None:
                raise HTTPException(
                    status_code=500,
                    detail='L\'agent n\'a pas pu générer de rapport. Vérifiez les logs pour plus de détails.',
                )
            
            # Détecter les rapports de fallback (agent interrompu)
            if _is_fallback_report(result):
                # Retourner le rapport mais avec un code 206 (Partial Content) et un warning
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=206,
                    content={
                        'report': result.model_dump() if hasattr(result, 'model_dump') else result,
                        'warning': 'L\'agent a été interrompu avant de finaliser l\'extraction. Le rapport contient des données partielles ou un rapport de fallback.',
                        'success': False,
                        'message': 'Vérifiez les logs du serveur pour plus de détails sur l\'échec de l\'extraction.',
                    },
                    headers={'X-Agent-Status': 'partial-fallback'},
                )
            
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'Erreur lors de l\'exécution: {str(exc)}') from exc

    return run_agent


def _is_fallback_report(report: Any) -> bool:
    """Détecte si un rapport est un rapport de fallback (agent interrompu).
    
    Les rapports de fallback ont des caractéristiques reconnaissables :
    - Nom "Informations indisponibles" 
    - Messages dans short_notes indiquant une interruption
    - Liste vide ou avec un seul élément placeholder
    """
    if not hasattr(report, 'model_dump'):
        return False
    
    report_dict = report.model_dump()
    
    # Vérifier pour StartupListingReport
    if 'startups' in report_dict:
        startups = report_dict.get('startups', [])
        if len(startups) == 1:
            startup = startups[0]
            if startup.get('name') == 'Informations indisponibles':
                short_notes = startup.get('short_notes', [])
                if any('interrompu' in note.lower() or 'fallback' in note.lower() for note in short_notes):
                    return True
    
    # Vérifier pour ProductResearchReport
    if 'company' in report_dict and 'products' in report_dict:
        company = report_dict.get('company', {})
        products = report_dict.get('products', [])
        if len(products) == 1:
            product = products[0]
            if product.get('product_name') == 'Informations indisponibles':
                other_facts = company.get('other_facts', [])
                if any('interrompu' in fact.lower() for fact in other_facts):
                    return True
    
    # Vérifier pour ProductHuntLeaderboardReport
    if 'products' in report_dict:
        products = report_dict.get('products', [])
        if len(products) == 1:
            product = products[0]
            if product.get('name') == 'Informations indisponibles':
                return True
    
    # Vérifier pour StartupExtractionReport
    if 'startups' in report_dict:
        startups = report_dict.get('startups', [])
        if len(startups) == 1:
            startup = startups[0]
            if startup.get('name') == 'Informations indisponibles':
                return True
    
    return False


def _get_examples_for_agent(agent_name: str, input_class: Type[BaseModel]) -> Dict[str, Dict[str, Any]]:
    """Génère des exemples pour un agent donné."""
    examples: Dict[str, Dict[str, Any]] = {}

    if agent_name == 'product_research':
        examples['product_hunt'] = {
            'summary': 'Recherche sur Product Hunt',
            'description': 'Exemple d\'extraction depuis une page Product Hunt avec tous les paramètres',
            'value': {
                'url': 'https://www.producthunt.com/posts/example-product',
                'max_products': 3,
                'output_path': 'product_research_report.json',
            },
        }
        examples['betalist'] = {
            'summary': 'Recherche sur BetaList',
            'description': 'Exemple d\'extraction depuis une page BetaList avec max_products personnalisé',
            'value': {
                'url': 'https://betalist.com/startups/example',
                'max_products': 5,
                'output_path': 'betalist_research.json',
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal (URL uniquement)',
            'description': 'Exemple avec seulement l\'URL requise, les autres paramètres utilisent les valeurs par défaut',
            'value': {
                'url': 'https://www.producthunt.com/posts/my-product',
            },
        }
    elif agent_name == 'startup_listing':
        examples['product_hunt_listing'] = {
            'summary': 'Listing Product Hunt complet',
            'description': 'Extraire une liste de startups depuis Product Hunt avec tous les paramètres',
            'value': {
                'url': 'https://www.producthunt.com/topics/startups',
                'max_startups': 50,
                'output_path': 'startup_listings.json',
            },
        }
        examples['betalist_listing'] = {
            'summary': 'Listing BetaList',
            'description': 'Extraire toutes les startups depuis BetaList (max_startups élevé)',
            'value': {
                'url': 'https://betalist.com/startups',
                'max_startups': 1000,
                'output_path': 'betalist_all.json',
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal',
            'description': 'Exemple avec seulement l\'URL requise',
            'value': {
                'url': 'https://www.producthunt.com/topics/ai',
            },
        }
    elif agent_name == 'universal_startup_extractor':
        examples['universal_extraction'] = {
            'summary': 'Extraction universelle complète',
            'description': 'Extraire toutes les startups d\'un site web avec limite personnalisée',
            'value': {
                'url': 'https://example.com/startups',
                'max_startup': 100,
                'output_path': 'extracted_startups.json',
            },
        }
        examples['extraction_illimitée'] = {
            'summary': 'Extraction illimitée',
            'description': 'Extraire toutes les startups sans limite (max_startup élevé)',
            'value': {
                'url': 'https://example.com/startups-directory',
                'max_startup': 100000,
                'output_path': 'all_startups.json',
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal',
            'description': 'Exemple avec seulement l\'URL requise',
            'value': {
                'url': 'https://example.com/startups',
            },
        }
    elif agent_name == 'product_hunt_leaderboard':
        examples['leaderboard_date'] = {
            'summary': 'Leaderboard avec date spécifique',
            'description': 'Extraire le leaderboard Product Hunt pour une date donnée (format YYYY-MM-DD)',
            'value': {
                'date': '2025-01-15',
                'max_products': 1000,
                'output_path': 'product_hunt_leaderboard.json',
            },
        }
        examples['leaderboard_aujourd_hui'] = {
            'summary': 'Leaderboard d\'aujourd\'hui',
            'description': 'Extraire le leaderboard du jour actuel',
            'value': {
                'date': '2025-01-20',
                'max_products': 500,
            },
        }
        examples['leaderboard_format_alternatif'] = {
            'summary': 'Date avec format alternatif',
            'description': 'Le format YYYY/MM/DD est aussi accepté',
            'value': {
                'date': '2025/01/15',
                'max_products': 1000,
            },
        }
    elif agent_name == 'futuretools_extractor':
        examples['newly_added'] = {
            'summary': 'Page newly-added (recommandé)',
            'description': 'Extraire depuis la page newly-added de FutureTools (URL par défaut)',
            'value': {
                'url': 'https://www.futuretools.io/newly-added',
                'max_tools': 1000,
                'output_path': 'futuretools_tools.json',
            },
        }
        examples['page_principale'] = {
            'summary': 'Page principale',
            'description': 'Extraire depuis la page principale de FutureTools',
            'value': {
                'url': 'https://www.futuretools.io/',
                'max_tools': 500,
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal',
            'description': 'Exemple avec seulement l\'URL requise (utilise les valeurs par défaut)',
            'value': {
                'url': 'https://www.futuretools.io/newly-added',
            },
        }
    elif agent_name == 'airtable_extractor':
        examples['shared_view'] = {
            'summary': 'Vue partagée Airtable',
            'description': 'Extraire les données depuis un lien Airtable partagé',
            'value': {
                'url': 'https://airtable.com/appXXXXXXXXXXXXXX/tblYYYYYYYYYYYYYY/viwZZZZZZZZZZZZZZ',
            },
        }
        examples['api_endpoint'] = {
            'summary': 'Endpoint API Airtable',
            'description': 'Extraire les données depuis un endpoint API Airtable',
            'value': {
                'url': 'https://api.airtable.com/v0/appXXXXXXXXXXXXXX/tblYYYYYYYYYYYYYY',
            },
        }
    elif agent_name == 'appsumo_hot_extractor':
        examples['whats_hot'] = {
            'summary': 'Collection What\'s Hot',
            'description': 'Extraire les produits tendance depuis AppSumo What\'s Hot',
            'value': {
                'url': 'https://appsumo.com/collections/whats-hot/',
                'max_products': 200,
                'output_path': 'appsumo_hot_products.json',
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal',
            'description': 'Exemple avec seulement l\'URL (utilise les valeurs par défaut)',
            'value': {
                'url': 'https://appsumo.com/collections/whats-hot/',
            },
        }
        examples['grande_extraction'] = {
            'summary': 'Grande extraction',
            'description': 'Extraire jusqu\'à 2000 produits',
            'value': {
                'url': 'https://appsumo.com/collections/whats-hot/',
                'max_products': 2000,
                'output_path': 'appsumo_hot_all.json',
            },
        }
    elif agent_name == 'appsumo_new_extractor':
        examples['new_arrivals'] = {
            'summary': 'Collection New Arrivals',
            'description': 'Extraire les nouveaux produits depuis AppSumo New Arrivals',
            'value': {
                'url': 'https://appsumo.com/collections/new/',
                'max_products': 200,
                'output_path': 'appsumo_new_products.json',
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal',
            'description': 'Exemple avec seulement l\'URL (utilise les valeurs par défaut)',
            'value': {
                'url': 'https://appsumo.com/collections/new/',
            },
        }
        examples['grande_extraction'] = {
            'summary': 'Grande extraction',
            'description': 'Extraire jusqu\'à 2000 nouveaux produits',
            'value': {
                'url': 'https://appsumo.com/collections/new/',
                'max_products': 2000,
                'output_path': 'appsumo_new_all.json',
            },
        }
    elif agent_name == 'betalist_extractor':
        examples['recent_startups'] = {
            'summary': 'Startups récentes',
            'description': 'Extraire les startups des 3 derniers jours depuis BetaList',
            'value': {
                'url': 'https://betalist.com/',
                'last_days': 3,
                'max_startups': 200,
                'output_path': 'betalist_recent.json',
            },
        }
        examples['derniere_semaine'] = {
            'summary': 'Dernière semaine',
            'description': 'Extraire les startups de la dernière semaine',
            'value': {
                'url': 'https://betalist.com/',
                'last_days': 7,
                'max_startups': 500,
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal',
            'description': 'Exemple avec seulement l\'URL (utilise les valeurs par défaut)',
            'value': {
                'url': 'https://betalist.com/',
            },
        }
    elif agent_name == 'stationf_companies_extractor':
        examples['sans_auth'] = {
            'summary': 'Sans authentification',
            'description': 'Extraire les entreprises sans authentification (si la page est accessible)',
            'value': {
                'url': 'https://hal2.stationf.co/companies',
                'max_companies': 1000,
                'output_path': 'stationf_companies.json',
            },
        }
        examples['avec_auth'] = {
            'summary': 'Avec authentification',
            'description': 'Extraire les entreprises avec authentification (email et mot de passe)',
            'value': {
                'url': 'https://hal2.stationf.co/companies',
                'max_companies': 1000,
                'output_path': 'stationf_companies.json',
                'email': 'votre@email.com',
                'password': 'votre_mot_de_passe',
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal',
            'description': 'Exemple avec seulement l\'URL (utilise les valeurs par défaut)',
            'value': {
                'url': 'https://hal2.stationf.co/companies',
            },
        }
    elif agent_name == 'zone_secure_startups_extractor':
        examples['extraction_complete'] = {
            'summary': 'Extraction complète',
            'description': 'Extraire toutes les startups depuis Zone Secure (jusqu\'à 10000)',
            'value': {
                'url': 'https://fr.zone-secure.net/20412/2540033/#page=1',
                'max_startups': 10000,
                'output_path': 'zone_secure_startups.json',
            },
        }
        examples['extraction_limitee'] = {
            'summary': 'Extraction limitée',
            'description': 'Extraire un nombre limité de startups',
            'value': {
                'url': 'https://fr.zone-secure.net/20412/2540033/#page=1',
                'max_startups': 100,
                'output_path': 'zone_secure_limited.json',
            },
        }
        examples['minimal'] = {
            'summary': 'Exemple minimal',
            'description': 'Exemple avec seulement l\'URL (utilise les valeurs par défaut)',
            'value': {
                'url': 'https://fr.zone-secure.net/20412/2540033/#page=1',
            },
        }

    return examples


# Création dynamique des endpoints pour chaque agent
for agent_name, agent_info in registry.list_agents().items():
    _create_agent_endpoint(
        agent_name=agent_name,
        input_class=agent_info['input_class'],
        output_class=agent_info['output_class'],
        run_function=agent_info['run_function'],
        description=agent_info['description'],
    )


@app.get(
    '/health',
    summary='Vérifier que l\'API répond',
    description='Endpoint de santé pour vérifier que l\'API est opérationnelle.',
    responses={
        200: {
            'description': 'API opérationnelle',
            'content': {
                'application/json': {
                    'example': {'status': 'ok', 'agents_count': len(registry.list_agents())}
                }
            },
        },
    },
)
def healthcheck() -> Dict[str, Any]:
    """Endpoint minimal pour surveiller l'API.

    **Exemple d'utilisation :**

    ```bash
    # Avec curl
    curl http://localhost:8000/health

    # Avec Python requests
    import requests
    response = requests.get('http://localhost:8000/health')
    print(response.json())  # {"status": "ok", "agents_count": 4}
    ```
    """
    return {'status': 'ok', 'agents_count': len(registry.list_agents())}
