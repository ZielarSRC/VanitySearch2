; VanitySearchBoost.nsi
!include "MUI2.nsh"

Name "VanitySearchBoost"
OutFile "VanitySearchBoost_Setup.exe"
InstallDir "$PROGRAMFILES64\VanitySearchBoost"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Main"
    SetOutPath $INSTDIR
    File "build\Release\VanitySearchBoost.exe"
    File "lib\OpenCL.dll"
    
    CreateShortCut "$SMPROGRAMS\VanitySearchBoost.lnk" "$INSTDIR\VanitySearchBoost.exe"
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VanitySearchBoost" \
        "DisplayName" "VanitySearchBoost"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VanitySearchBoost" \
        "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\VanitySearchBoost.exe"
    Delete "$INSTDIR\OpenCL.dll"
    Delete "$INSTDIR\Uninstall.exe"
    Delete "$SMPROGRAMS\VanitySearchBoost.lnk"
    RMDir "$INSTDIR"
    
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VanitySearchBoost"
SectionEnd
