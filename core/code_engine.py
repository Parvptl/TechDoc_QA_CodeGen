import ast
import tempfile
import sys
import contextlib
import multiprocessing
import os
import time

ALLOWED_IMPORTS = {
    'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'scipy', 
    'math', 'statistics', 'collections', 'itertools', 'json', 'csv', 
    'warnings', 'os', 're', 'datetime', 'time', 'typing', 'functools', 'operator'
}

class CodeEngine:
    """Safely executes Data Science code snippets in a validated python sandbox."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def validate_code(self, source_code: str) -> bool:
        """Validates using AST. Checks for dangerous imports or functions."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split('.')[0]
                    if base_module not in ALLOWED_IMPORTS:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module not in ALLOWED_IMPORTS:
                        return False
            
            # Check forbidden functions/calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {'eval', 'exec', 'open', '__import__'}:
                        return False
        return True

    def _execute_in_process(self, source_code: str, q: multiprocessing.Queue):
        import io
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Setup captured stdout/stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                # Prepare isolated globals
                exec_globals = {
                    '__builtins__': {k: v for k, v in __builtins__.items() if k not in {'eval', 'exec', 'open', '__import__'}}
                }
                exec(source_code, exec_globals)
                
                # Check for plots
                figs = [plt.figure(n) for n in plt.get_fignums()]
                has_plot = len(figs) > 0
                plt.close('all')
                
                q.put({
                    'success': True,
                    'stdout': stdout.getvalue(),
                    'stderr': stderr.getvalue(),
                    'has_plot': has_plot,
                    'error': None
                })
            except Exception as e:
                q.put({
                    'success': False,
                    'stdout': stdout.getvalue(),
                    'stderr': stderr.getvalue(),
                    'has_plot': False,
                    'error': str(e)
                })

    def execute(self, source_code: str) -> dict:
        """Executes the code securely and returns execution outputs."""
        if not self.validate_code(source_code):
            return {
                'success': False,
                'stdout': '',
                'stderr': '',
                'error': 'Code Validation Failed: Snippet contains unwhitelisted imports or unsafe function calls.',
                'has_plot': False
            }

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._execute_in_process, args=(source_code, q))
        
        start_time = time.time()
        p.start()
        p.join(self.timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            return {
                'success': False,
                'stdout': '',
                'stderr': '',
                'error': f'Execution Timeout: Code ran longer than {self.timeout} seconds.',
                'has_plot': False
            }
        
        if not q.empty():
            return q.get()
        else:
            return {
                'success': False,
                'error': 'Sandbox Error: No output returned from execution process.',
                'stdout': '', 'stderr': '', 'has_plot': False
            }
