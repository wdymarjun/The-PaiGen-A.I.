# ==============================================================================
# PaiGen AI — Pure Android SDK APK Compiler & Signer (Zero-Gradle Fast Build)
# Uses Android SDK (aapt2, d8, zipalign, apksigner) + Android Studio JBR JDK
# ==============================================================================
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Join-Path $Root "app\src\main"
$BuildDir = Join-Path $Root "build"
$DistDir = Join-Path $Root "dist"

$SdkRoot = if ($env:ANDROID_SDK_ROOT) { $env:ANDROID_SDK_ROOT } else { Join-Path $env:LOCALAPPDATA "Android\Sdk" }
$BuildTools = Join-Path $SdkRoot "build-tools\36.0.0"
$PlatformJar = Join-Path $SdkRoot "platforms\android-37.0\android.jar"

$JbrBin = "C:\Program Files\Android\Android Studio\jbr\bin"
$Javac = Join-Path $JbrBin "javac.exe"
$Keytool = Join-Path $JbrBin "keytool.exe"
$Aapt2 = Join-Path $BuildTools "aapt2.exe"
$D8 = Join-Path $BuildTools "d8.bat"
$ZipAlign = Join-Path $BuildTools "zipalign.exe"
$ApkSigner = Join-Path $BuildTools "apksigner.bat"

foreach ($req in @($PlatformJar, $Javac, $Aapt2, $D8, $ZipAlign, $ApkSigner)) {
    if (-not (Test-Path $req)) {
        throw "Required Android build tool not found: $req"
    }
}

if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
New-Item -ItemType Directory -Force -Path $BuildDir, "$BuildDir\res_compiled", "$BuildDir\gen", "$BuildDir\classes", "$BuildDir\dex", $DistDir | Out-Null

$Manifest = Join-Path $AppDir "AndroidManifest.xml"
$ResDir = Join-Path $AppDir "res"
$AssetsDir = Join-Path $AppDir "assets"
$JavaRoot = Join-Path $AppDir "java"

Write-Host "[1/6] Compiling Android resources with aapt2..."
& $Aapt2 compile --dir $ResDir -o "$BuildDir\res.zip"
if ($LASTEXITCODE -ne 0) { throw "aapt2 compile failed" }

Write-Host "[2/6] Linking resources, assets, and generating R.java..."
$UnsignedApk = Join-Path $BuildDir "app-unsigned.apk"
& $Aapt2 link `
    -I $PlatformJar `
    --manifest $Manifest `
    -A $AssetsDir `
    --java "$BuildDir\gen" `
    -o $UnsignedApk `
    "$BuildDir\res.zip"
if ($LASTEXITCODE -ne 0) { throw "aapt2 link failed" }

Write-Host "[3/6] Compiling Java sources (target 11)..."
$JavaFiles = @(Get-ChildItem -Path $JavaRoot, "$BuildDir\gen" -Recurse -Filter "*.java" | ForEach-Object { $_.FullName })
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
& $Javac --release 11 -encoding UTF-8 -classpath $PlatformJar -d "$BuildDir\classes" $JavaFiles
if ($LASTEXITCODE -ne 0) { throw "javac compilation failed" }

Write-Host "[4/6] Translating bytecode to DEX with d8..."
$ClassFiles = @(Get-ChildItem -Path "$BuildDir\classes" -Recurse -Filter "*.class" | ForEach-Object { $_.FullName })
& $D8 --min-api 26 --lib $PlatformJar --output "$BuildDir\dex" $ClassFiles
if ($LASTEXITCODE -ne 0) { throw "d8 dex compilation failed" }

Write-Host "[5/6] Packaging classes.dex & aligning APK..."
# Inject classes.dex into APK zip archive cleanly via Python zipfile (preserves zip entries)
python -c "import zipfile; z=zipfile.ZipFile(r'$UnsignedApk','a',compression=zipfile.ZIP_DEFLATED); z.write(r'$BuildDir\dex\classes.dex','classes.dex'); z.close()"
if ($LASTEXITCODE -ne 0) { throw "Injecting classes.dex failed" }

$AlignedApk = Join-Path $BuildDir "app-aligned.apk"
& $ZipAlign -f -p 4 $UnsignedApk $AlignedApk
if ($LASTEXITCODE -ne 0) { throw "zipalign failed" }

Write-Host "[6/6] Signing APK with local keystore..."
$Keystore = Join-Path $Root "paigen-debug.keystore"
if (-not (Test-Path $Keystore)) {
    & $Keytool -genkeypair -v -keystore $Keystore -alias paigen -keyalg RSA -keysize 2048 -validity 10000 -storepass paigen123 -keypass paigen123 -dname "CN=Arjun Singh, OU=PaiGen AI, O=PaiGen, L=India, C=IN" | Out-Null
}

$FinalApk = Join-Path $DistDir "PaiGen-AI.apk"
& $ApkSigner sign --ks $Keystore --ks-pass pass:paigen123 --key-pass pass:paigen123 --ks-key-alias paigen --out $FinalApk $AlignedApk
if ($LASTEXITCODE -ne 0) { throw "apksigner sign failed" }

& $ApkSigner verify --print-certs $FinalApk
if ($LASTEXITCODE -ne 0) { throw "apksigner verify failed" }

$SizeMB = [math]::Round((Get-Item $FinalApk).Length / 1KB, 1)
Write-Host "`n[+] SUCCESS! Built & verified signed APK: $FinalApk ($SizeMB KB)"
