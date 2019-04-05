/******************************************************************************
Copyright:  BICI2
Created:    28:3:2016 16:05
Filename:   export.h
Author:     Pingjun Chen

Purpose:    for dll export and import
******************************************************************************/


#ifndef MUSCLEMINER_EXPORT_H_
#define MUSCLEMINER_MUSCLE_SEG_EXPORT_H_

#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__)
#  if defined( MUSCLEMINER_LIBRARY_STATIC )
#    define MUSCLEMINER_EXPORT
#  elif defined( MUSCLEMINER_MUSCLE_SEG_LIBRARY )
#    define MUSCLEMINER_EXPORT   __declspec(dllexport)
#  else
#    define MUSCLEMINER_EXPORT   __declspec(dllimport)
#  endif
#else
#  define MUSCLEMINER_EXPORT
#endif

#endif // MUSCLEMINER_EXPORT_H_
