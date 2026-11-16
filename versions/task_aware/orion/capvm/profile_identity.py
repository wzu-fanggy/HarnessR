"""从已登记的本地工作进程自动生成版本身份，不启动资源或执行模型工厂。"""

from pathlib import Path

import hashlib,importlib.metadata,inspect,os,platform,sys

from .resource_profiles import profile_fingerprint



class ProcessIdentityResolver:

    def __init__(self, *, packages=('numpy','scipy','pandas','scikit-learn','joblib')):

        self.packages=packages;self._hashes={}

    def _digest(self,path):

        path=Path(path).resolve();stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)

        if key not in self._hashes:

            digest=hashlib.sha256()

            with path.open('rb') as f:

                for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)

            self._hashes={k:v for k,v in self._hashes.items() if k[0]!=str(path)}

            self._hashes[key]=digest.hexdigest()

        return self._hashes[key]

    def resolve(self,capability):

        adapter=capability.adapter

        worker=getattr(adapter,'worker_script',None)

        if worker is None:raise ValueError('resource has no owned worker identity')

        module=Path(inspect.getfile(type(adapter))).resolve()

        # 同目录本地实现一并指纹化，保守失效优于遗漏实际计算内核。

        paths=set(module.parent.glob('*.py'))|{module,Path(worker).resolve()}

        model=getattr(adapter,'model_path',None)

        if getattr(adapter,'model_path_factory',None) is not None and model is None:

            raise ValueError('model artifact unresolved; factory is not executed for profiling')

        if model is not None:paths.add(Path(model).resolve())

        bundle=getattr(adapter,'bundle_dir',None)

        if hasattr(adapter,'bundle_dir'):

            if bundle is None:bundle=(getattr(adapter,'ready_metadata',None) or {}).get('bundle_dir')

            if bundle is None:raise ValueError('bundle identity unresolved')

            paths.update(p for p in Path(bundle).rglob('*') if p.is_file())

        artifacts={str(p):self._digest(p) for p in sorted(paths)}

        versions={}

        for name in self.packages:

            try:versions[name]=importlib.metadata.version(name)

            except importlib.metadata.PackageNotFoundError:versions[name]='unavailable'

        environment={'schema':1,'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),

            'processor':platform.processor(),'device':'cpu','packages':versions,

            'threads':{k:os.environ.get(k,'library_default') for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},

            'measurement_scope':'owned_single_worker','input_class':'startup_only'}

        identity={'artifacts':artifacts,'environment':environment}

        return profile_fingerprint(**identity),identity

