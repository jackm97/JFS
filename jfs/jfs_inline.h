#ifdef JFS_INLINE
#undef JFS_INLINE
#endif

#ifndef JFS_STATIC
#  define JFS_INLINE inline
#else
#  define JFS_INLINE
#endif