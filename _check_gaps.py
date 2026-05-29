"""Check available alt-memory features."""
from alt_memory.config import AltMemoryConfig
import os

# Check hooks read format
c = AltMemoryConfig()
print("hook_silent_save source: file_config.get('hook_silent_save', True)")

c2 = AltMemoryConfig()
c2._file_config = {"hook_silent_save": True, "hooks_auto_save": False}
print(f"  flat format: silent_save={c2.hook_silent_save}, auto_save={c2.hooks_auto_save}")

# alt-memory uses flat format
c3 = AltMemoryConfig()
c3._file_config = {"hooks": {"silent_save": True, "auto_save": False}}
print(f"  nested hooks: silent_save={c3.hook_silent_save}, auto_save={c3.hooks_auto_save}")

# Check ALT_MEMORY_PATH
os.environ["ALT_MEMORY_PATH"] = "/tmp/test"
c4 = AltMemoryConfig()
print(f"dim_path with ALT_MEMORY_PATH: {c4.dim_path}")
del os.environ["ALT_MEMORY_PATH"]

# ALT_TOPIC_TUNNEL_MIN_COUNT
os.environ["ALT_TOPIC_TUNNEL_MIN_COUNT"] = "3"
c5 = AltMemoryConfig()
print(f"topic_tunnel_min_count with ALT_TOPIC_TUNNEL_MIN_COUNT: {c5.topic_tunnel_min_count}")
del os.environ["ALT_TOPIC_TUNNEL_MIN_COUNT"]

# ALT_ENTITY_LANGUAGES
os.environ["ALT_ENTITY_LANGUAGES"] = "en,fr"
c6 = AltMemoryConfig()
print(f"entity_languages with ALT_ENTITY_LANGUAGES: {c6.entity_languages}")
del os.environ["ALT_ENTITY_LANGUAGES"]

# ALT_HOOKS_AUTO_SAVE
os.environ["ALT_HOOKS_AUTO_SAVE"] = "false"
c7 = AltMemoryConfig()
print(f"hooks_auto_save with ALT_HOOKS_AUTO_SAVE: {c7.hooks_auto_save}")
del os.environ["ALT_HOOKS_AUTO_SAVE"]

print()

# Check if config file keys are nested or flat
print("Config hooks key format check:")
print(f"  hook_silent_save reads from: 'hook_silent_save' (flat) or 'hooks.silent_save' (nested)")
print(f"  alt-memory config uses: hooks: {{silent_save: ..., auto_save: ...}}")

# Check migrate.py compat
print()
print("Checking migration support...")
try:
    from alt_memory import migrate
    print("  migrate.py: available")
except ImportError:
    print("  migrate.py: MISSING")

# Check repair features
print()
print("Checking repair features...")
from alt_memory import repair
print(f"  repair functions: {[f for f in dir(repair) if not f.startswith('_')]}")

# Check exporter
print()
print("Checking exporter...")
try:
    from alt_memory import exporter
    print("  exporter.py: available")
except ImportError:
    print("  exporter.py: MISSING")

# Check for benchmark support
print()
print("Checking benchmarks...")
try:
    import alt_memory.benchmarks
    print("  benchmarks: available")
except ImportError:
    print("  benchmarks: MISSING")

# Check for embedding_model config
print()
print("Checking embedding config...")
print(f"  AltMemoryConfig has embedding_device: {hasattr(AltMemoryConfig, 'embedding_device')}")
print(f"  AltMemoryConfig has embedding_model: {hasattr(AltMemoryConfig, 'embedding_model')}")
print(f"  AltMemoryConfig has collection_name: {hasattr(AltMemoryConfig, 'collection_name')}")
print(f"  AltMemoryConfig has set_hook_setting: {hasattr(AltMemoryConfig, 'set_hook_setting')}")
print(f"  AltMemoryConfig has set_embedding_model: {hasattr(AltMemoryConfig, 'set_embedding_model')}")
print(f"  AltMemoryConfig has save_people_map: {hasattr(AltMemoryConfig, 'save_people_map')}")
