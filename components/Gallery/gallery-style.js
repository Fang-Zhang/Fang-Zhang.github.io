import { makeStyles } from 'tss-react/mui';

const galleryStyles = makeStyles({ uniqId: 'gallery' })((theme, _params, classes) => ({
  root: {
    position: 'relative'
  },
  loaded: {},
  selected: {},
  filter: {
    position: 'relative',
    zIndex: 20,
    display: 'flex',
    overflow: 'auto',
    width: '100%',
    padding: theme.spacing(3, 0),
    [theme.breakpoints.down('md')]: {
      paddingTop: 0
    },
    '& > button': {
      background: 'transparent',
      borderRadius: 40,
      marginRight: theme.spacing(),
      border: '1px solid rgba(0, 0, 0, 0.87)',
      color: theme.palette.text.primary,
      whiteSpace: 'nowrap',
      minWidth: 100,
      [theme.breakpoints.up('sm')]: {
        padding: theme.spacing(1, 4),
        marginRight: theme.spacing(3),
      },
      [`&.${classes.selected}`]: {
        background: 'rgba(0, 0, 0, 0.87)',
        color: theme.palette.common.white
      }
    }
  },
  massonry: {
    columns: `${3} 300px`,
    columnGap: theme.spacing(4),
  },
  item: {
    marginBottom: theme.spacing(4),
    breakInside: 'avoid',
    opacity: 0,
    position: 'relative',
    paddingTop: 20,
    [`&.${classes.loaded}`]: {
      transition: 'all 0.3s ease-out',
      opacity: 1,
      paddingTop: 0,
    }
  },
  itemCarousel: {
    position: 'relative'
  }
}));

// TODO jss-to-tss-react codemod: usages of this hook outside of this file will not be converted.
export default galleryStyles;
