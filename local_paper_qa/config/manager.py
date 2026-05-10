import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class APIConfig:
    """Configuration for academic APIs."""
    semantic_scholar_api_key: Optional[str] = None
    crossref_api_key: Optional[str] = None
    exa_api_key: Optional[str] = None
    rate_limit_per_hour: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_enhanced_lineage: bool = True


@dataclass
class SystemConfig:
    """System configuration."""
    papers_dir: str = "papers"
    index_dir: str = ".research_index"
    lineage_dir: str = ".enhanced_lineage"
    chat_url: str = "http://100.67.104.58:8001/v1"
    embedding_url: str = "http://100.67.104.58:8003/v1"
    chat_model: str = "unsloth/Qwen3.6"
    embedding_model: str = "unsloth/Qwen3.6"
    max_chunk_size: int = 2000
    max_chunks_per_page: int = 10


@dataclass
class AppConfig:
    """Main application configuration."""
    api: APIConfig
    system: SystemConfig
    debug: bool = False
    log_level: str = "INFO"


class ConfigManager:
    """Manages application configuration from environment variables and config files."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from environment variables and config file."""
        # Load from config file if it exists
        file_config = self._load_from_file()
        
        # Load from environment variables (override file config)
        api_config = self._load_api_config(file_config)
        system_config = self._load_system_config(file_config)
        
        return AppConfig(
            api=api_config,
            system=system_config,
            debug=os.environ.get("DEBUG", "false").lower() == "true",
            log_level=os.environ.get("LOG_LEVEL", "INFO")
        )
    
    def _load_from_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file."""
        config_path = Path(self.config_file)
        if not config_path.exists():
            return None
        
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return None
    
    def _load_api_config(self, file_config: Optional[Dict[str, Any]]) -> APIConfig:
        """Load API configuration from file and environment."""
        # Start with file config if available
        if file_config and 'api' in file_config:
            api_data = file_config['api']
        else:
            api_data = {}
        
        # Override with environment variables
        return APIConfig(
            semantic_scholar_api_key=api_data.get('semantic_scholar_api_key') or os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
            crossref_api_key=api_data.get('crossref_api_key') or os.environ.get("CROSSREF_API_KEY"),
            exa_api_key=api_data.get('exa_api_key') or os.environ.get("EXA_API_KEY"),
            rate_limit_per_hour=api_data.get('rate_limit_per_hour') or int(os.environ.get("RATE_LIMIT_PER_HOUR", "1000")),
            timeout_seconds=api_data.get('timeout_seconds') or int(os.environ.get("TIMEOUT_SECONDS", "30")),
            max_retries=api_data.get('max_retries') or int(os.environ.get("MAX_RETRIES", "3")),
            enable_enhanced_lineage=api_data.get('enable_enhanced_lineage') or os.environ.get("ENABLE_ENHANCED_LINEAGE", "true").lower() == "true"
        )
    
    def _load_system_config(self, file_config: Optional[Dict[str, Any]]) -> SystemConfig:
        """Load system configuration from file and environment."""
        # Start with file config if available
        if file_config and 'system' in file_config:
            system_data = file_config['system']
        else:
            system_data = {}
        
        # Override with environment variables
        return SystemConfig(
            papers_dir=system_data.get('papers_dir') or os.environ.get("PAPERS_DIR", "papers"),
            index_dir=system_data.get('index_dir') or os.environ.get("INDEX_DIR", ".research_index"),
            lineage_dir=system_data.get('lineage_dir') or os.environ.get("LINEAGE_DIR", ".enhanced_lineage"),
            chat_url=system_data.get('chat_url') or os.environ.get("LOCAL_PAPER_QA_CHAT_URL", "http://100.67.104.58:8001/v1"),
            embedding_url=system_data.get('embedding_url') or os.environ.get("LOCAL_PAPER_QA_EMBEDDING_URL", "http://100.67.104.58:8003/v1"),
            chat_model=system_data.get('chat_model') or os.environ.get("LOCAL_PAPER_QA_CHAT_MODEL", "unsloth/Qwen3.6"),
            embedding_model=system_data.get('embedding_model') or os.environ.get("LOCAL_PAPER_QA_EMBEDDING_MODEL", "unsloth/Qwen3.6"),
            max_chunk_size=system_data.get('max_chunk_size') or int(os.environ.get("MAX_CHUNK_SIZE", "2000")),
            max_chunks_per_page=system_data.get('max_chunks_per_page') or int(os.environ.get("MAX_CHUNKS_PER_PAGE", "10"))
        )
    
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            import json
            config_dict = asdict(self.config)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service."""
        if service == "semantic_scholar":
            return self.config.api.semantic_scholar_api_key
        elif service == "crossref":
            return self.config.api.crossref_api_key
        elif service == "exa":
            return self.config.api.exa_api_key
        return None
    
    def has_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are available."""
        return {
            "semantic_scholar": bool(self.config.api.semantic_scholar_api_key),
            "crossref": bool(self.config.api.crossref_api_key),
            "exa": bool(self.config.api.exa_api_key)
        }
    
    def get_available_apis(self) -> list[str]:
        """Get list of available APIs based on API keys."""
        available = []
        if self.config.api.semantic_scholar_api_key:
            available.append("semantic_scholar")
        if self.config.api.crossref_api_key:
            available.append("crossref")
        if self.config.api.exa_api_key:
            available.append("exa")
        return available
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif hasattr(self.config.api, key):
                setattr(self.config.api, key, value)
            elif hasattr(self.config.system, key):
                setattr(self.config.system, key, value)
    
    def create_sample_config(self) -> None:
        """Create a sample configuration file."""
        sample_config = {
            "api": {
                "semantic_scholar_api_key": "your_semantic_scholar_api_key_here",
                "crossref_api_key": "your_crossref_api_key_here",
                "exa_api_key": "your_exa_api_key_here",
                "rate_limit_per_hour": 1000,
                "timeout_seconds": 30,
                "max_retries": 3,
                "enable_enhanced_lineage": True
            },
            "system": {
                "papers_dir": "papers",
                "index_dir": ".research_index",
                "lineage_dir": ".enhanced_lineage",
                "chat_url": "http://100.67.104.58:8001/v1",
                "embedding_url": "http://100.67.104.58:8003/v1",
                "chat_model": "unsloth/Qwen3.6",
                "embedding_model": "unsloth/Qwen3.6",
                "max_chunk_size": 2000,
                "max_chunks_per_page": 10
            },
            "debug": False,
            "log_level": "INFO"
        }
        
        try:
            import json
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, indent=2, ensure_ascii=False)
            print(f"Sample configuration created at {self.config_file}")
        except Exception as e:
            print(f"Error creating sample config: {e}")